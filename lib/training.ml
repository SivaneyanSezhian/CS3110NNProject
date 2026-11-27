type tensor = float Tensor.tensor

type forward = Autograd.t list -> tensor -> Autograd.t

type loss_fn = Autograd.t -> tensor -> Autograd.t

type history = {
  train_losses : float list;
  validation_losses : float list option;
}

let check_learning_rate learning_rate =
  if learning_rate <= 0. || not (Float.is_finite learning_rate) then
    invalid_arg "learning_rate must be a positive finite number"

let update_parameters learning_rate params =
  List.map
    (fun p ->
      let weights = Autograd.value p in
      let gradient = Autograd.grad p in
      Tensor.pointwise2
        (fun w g -> w -. learning_rate *. g)
        weights gradient)
    params

let train_step ~learning_rate ~forward ~loss_fn ~parameters ~inputs ~targets =
  check_learning_rate learning_rate;
  (match parameters with
  | [] -> invalid_arg "train_step requires at least one parameter"
  | _ -> ());
  let nodes = List.map Autograd.make parameters in
  let prediction = forward nodes inputs in
  let loss = loss_fn prediction targets in
  Autograd.backward loss;
  let updated = update_parameters learning_rate nodes in
  let loss_value = Tensor.lookup (Autograd.value loss) [] in
  (updated, loss_value)

let predict ~forward ~parameters ~inputs =
  let nodes = List.map Autograd.make parameters in
  Autograd.value (forward nodes inputs)

let evaluate ~forward ~loss_fn ~parameters ~inputs ~targets =
  let nodes = List.map Autograd.make parameters in
  let prediction = forward nodes inputs in
  let loss = loss_fn prediction targets in
  Tensor.lookup (Autograd.value loss) []

let train ?validation ~learning_rate ~epochs ~forward ~loss_fn ~parameters ~inputs
    ~targets () =
  if epochs <= 0 then invalid_arg "epochs must be positive";
  check_learning_rate learning_rate;
  let rec loop remaining current train_rev validation_rev =
    if remaining = 0 then
      let validation_losses =
        match validation with
        | None -> None
        | Some _ -> Some (List.rev validation_rev)
      in
      (current, { train_losses = List.rev train_rev; validation_losses })
    else
      let updated, _ =
        train_step ~learning_rate ~forward ~loss_fn ~parameters:current ~inputs ~targets
      in
      (* Record losses of the updated model, after this epoch's weight update. *)
      let train_loss =
        evaluate ~forward ~loss_fn ~parameters:updated ~inputs ~targets
      in
      let validation_rev =
        match validation with
        | None -> validation_rev
        | Some (validation_inputs, validation_targets) ->
            evaluate ~forward ~loss_fn ~parameters:updated ~inputs:validation_inputs
              ~targets:validation_targets
            :: validation_rev
      in
      loop (remaining - 1) updated (train_loss :: train_rev) validation_rev
  in
  loop epochs parameters [] []
