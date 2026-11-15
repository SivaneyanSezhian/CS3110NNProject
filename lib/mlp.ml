module T = Tensor
module A = Autograd

type activation =
  | Identity
  | Relu
  | Sigmoid
  | Tanh

type layer = {
  units : int;
  activation : activation;
}

type t = {
  input_dim : int;
  layers : layer list;
}
(** Abstraction function: [{input_dim; layers}] is the network that reads
    [input_dim] features and applies [layers] in order, each layer reading the
    output of the one before it.

    Representation invariant: [input_dim > 0], [layers] is nonempty, and every
    layer has [units > 0]. *)

let create ~input_dim layers =
  if input_dim <= 0 then invalid_arg "Mlp.create: input_dim must be positive";
  if layers = [] then invalid_arg "Mlp.create: need at least one layer";
  if List.exists (fun l -> l.units <= 0) layers then
    invalid_arg "Mlp.create: layer sizes must be positive";
  { input_dim; layers }

let input_dim net = net.input_dim
let layers net = net.layers

let output_dim net =
  match List.rev net.layers with
  | last :: _ -> last.units
  | [] -> failwith "Mlp.output_dim: invariant violated"

(** [layer_shapes net] pairs each layer with its input width, in order. *)
let layer_shapes net =
  let rec go inputs = function
    | [] -> []
    | l :: rest -> (inputs, l) :: go l.units rest
  in
  go net.input_dim net.layers

let num_parameters net =
  List.fold_left
    (fun acc (inputs, l) -> acc + (inputs * l.units) + l.units)
    0 (layer_shapes net)

(** [init_scale a] is the target weight variance times the input width for a
    layer with activation [a]. ReLU zeroes about half its inputs, so it gets
    twice the variance to keep the output scale steady across layers. *)
let init_scale = function
  | Relu -> 2.
  | Identity | Sigmoid | Tanh -> 1.

let init_parameters ?seed net =
  let state =
    match seed with
    | Some s -> Random.State.make [| s |]
    | None -> Random.State.make_self_init ()
  in
  List.concat_map
    (fun (inputs, l) ->
      (* A uniform draw on [-r, r] has variance r^2 / 3, so this choice of r
         gives variance [init_scale / inputs]. *)
      let r = sqrt (3. *. init_scale l.activation /. float_of_int inputs) in
      let w =
        T.make_tensor [ inputs; l.units ] (fun _ ->
            Random.State.float state (2. *. r) -. r)
      in
      let b = T.zeroes [ 1; l.units ] in
      [ w; b ])
    (layer_shapes net)

(** [apply_activation a x] applies activation [a] to node [x]. *)
let apply_activation a x =
  match a with
  | Identity -> x
  | Relu -> A.relu x
  | Sigmoid -> A.sigmoid x
  | Tanh -> A.tanh x

let forward net params inputs =
  let batch =
    match T.get_dims inputs with
    | [ batch; width ] when batch > 0 && width = net.input_dim -> batch
    | _ -> invalid_arg "Mlp.forward: inputs must have shape [batch; input_dim]"
  in
  (* Autograd only adds tensors of equal shape. Multiplying a column of ones by
     the bias row copies it onto every row, and the backward pass of that
     product sums the bias gradient over the batch. *)
  let ones = A.make (T.ones [ batch; 1 ]) in
  let rec go x shapes params =
    match (shapes, params) with
    | [], [] -> x
    | (inputs, l) :: shapes, w :: b :: params ->
        if
          T.get_dims (A.value w) <> [ inputs; l.units ]
          || T.get_dims (A.value b) <> [ 1; l.units ]
        then
          invalid_arg "Mlp.forward: parameter shapes do not match the network";
        let z = A.add (A.matmul x w) (A.matmul ones b) in
        go (apply_activation l.activation z) shapes params
    | _ -> invalid_arg "Mlp.forward: wrong number of parameters"
  in
  go (A.make inputs) (layer_shapes net) params

let activation_of_string s =
  match String.lowercase_ascii (String.trim s) with
  | "identity" | "linear" -> Some Identity
  | "relu" -> Some Relu
  | "sigmoid" -> Some Sigmoid
  | "tanh" -> Some Tanh
  | _ -> None

let string_of_activation = function
  | Identity -> "identity"
  | Relu -> "relu"
  | Sigmoid -> "sigmoid"
  | Tanh -> "tanh"

let to_string net =
  let layer_lines =
    List.mapi
      (fun i (inputs, l) ->
        Printf.sprintf "layer %d: %d -> %d, %s" (i + 1) inputs l.units
          (string_of_activation l.activation))
      (layer_shapes net)
  in
  String.concat "\n"
    ((Printf.sprintf "input: %d" net.input_dim :: layer_lines)
    @ [ Printf.sprintf "parameters: %d" (num_parameters net) ])
