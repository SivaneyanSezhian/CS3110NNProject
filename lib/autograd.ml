module T = Tensor
module F = Function

type t = {
  value : float T.tensor;
  mutable grad : float T.tensor;
  mutable inputs : t list;
  mutable backward_fn : unit -> unit;
}
(** Abstraction function: Each record represents one graph node. [value] is its
    forward result, [grad] is its stored gradient, and [inputs] lists the
    operand nodes in argument order. Record identity determines node identity.

    Representation invariant: [value] and [grad] are well-formed tensors with
    the same shape. Input edges are acyclic because an operation creates a fresh
    result and points it at existing nodes. An input may appear more than once
    when an operation uses the same node in multiple argument positions. A leaf
    has empty [inputs] and an inert [backward_fn]. For an operation,
    [backward_fn] applies the local chain rule to [grad] and adds a contribution
    to the gradient of each differentiable operand. An operation installs both
    [inputs] and [backward_fn] before returning its result. *)

let make value =
  {
    value;
    grad = T.zeroes (T.get_dims value);
    inputs = [];
    backward_fn = (fun () -> ());
  }

(** [accumulate node contribution] adds one gradient contribution to [node].
    [contribution] must have the same shape as [node.value]. Addition preserves
    contributions from other uses of the same node. *)
let accumulate node contribution =
  node.grad <- T.pointwise2 ( +. ) node.grad contribution

let add a b =
  assert (T.same_dims a.value b.value);
  let value = T.pointwise2 ( +. ) a.value b.value in
  let out = make value in
  out.backward_fn <-
    (fun () ->
      accumulate a out.grad;
      accumulate b out.grad);
  out.inputs <- [ a; b ];
  out

let matmul a b =
  let a_dims = T.get_dims a.value in
  let b_dims = T.get_dims b.value in
  assert (List.length a_dims = 2 && List.length b_dims = 2);
  let fst = function
    | x :: _ -> x
    | _ -> failwith "Empty list"
  in
  let snd = function
    | _ :: y :: _ -> y
    | _ -> failwith "List has fewer than two elements"
  in
  assert (snd a_dims = fst b_dims);
  let value = T.matmul a.value b.value in
  let out = make value in
  out.backward_fn <-
    (fun () ->
      (* For C = AB and upstream gradient G, the two contributions are G times B
         transposed and A transposed times G. *)
      let a_t = T.swap_dims a.value 0 1 in
      let b_t = T.swap_dims b.value 0 1 in
      let grad_a = T.matmul out.grad b_t in
      let grad_b = T.matmul a_t out.grad in
      accumulate a grad_a;
      accumulate b grad_b);
  out.inputs <- [ a; b ];
  out

(** [activation f x] lifts a scalar function and its derivative to a tensor
    operation. [f.grad_1] is evaluated at the original input, using the
    operation's chosen value at any nondifferentiable point. Since each output
    depends only on the matching input element, the backward rule multiplies the
    local and upstream gradients element-wise. *)
let activation (f : F.unary_function) x =
  let value = T.pointwise f.f_1 x.value in
  let out = make value in
  out.backward_fn <-
    (fun () ->
      let grad_act = T.pointwise f.grad_1 x.value in
      let grad_x = T.pointwise2 ( *. ) grad_act out.grad in
      accumulate x grad_x);
  out.inputs <- [ x ];
  out

let sigmoid = activation F.sig_f
let relu = activation F.relu_f
let tanh = activation F.tanh_f

let mse_loss pred target =
  assert (T.same_dims pred.value target);
  let error = T.pointwise2 ( -. ) pred.value target in
  let squared_error = T.pointwise (fun x -> x ** 2.) error in
  let sum = T.sum squared_error in
  let num = float_of_int (T.size squared_error) in
  let loss_value = sum /. num in
  let out = make (T.make_tensor [] (fun _ -> loss_value)) in
  out.backward_fn <-
    (fun () ->
      (* Reuse the forward residuals. The upstream gradient is scalar, so it
         scales every element of the local gradient. *)
      let upstream = T.lookup out.grad [] in
      let grad_pred =
        T.pointwise (fun x -> upstream *. (2. *. x /. num)) error
      in
      accumulate pred grad_pred);
  out.inputs <- [ pred ];
  out

let backward loss =
  assert (T.get_dims loss.value = []);
  (* Prepending on return from DFS puts each node before its operands. A shared
     operand runs once, after all its users have contributed to it. Physical
     identity keeps distinct records with equal values separate. *)
  let visited = ref [] in
  let ordering = ref [] in
  let rec visit node =
    if not (List.memq node !visited) then begin
      visited := node :: !visited;
      List.iter visit node.inputs;
      ordering := node :: !ordering
    end
  in

  visit loss;

  (* Clear gradients before any callback adds a contribution. The derivative of
     the scalar loss with respect to itself seeds the chain rule. *)
  List.iter
    (fun node -> node.grad <- T.zeroes (T.get_dims node.value))
    !ordering;

  loss.grad <- T.ones [];

  List.iter (fun node -> node.backward_fn ()) !ordering

let value x = x.value
let grad x = x.grad
