(** Training and inference with a supplied neural-network forward function.
    The forward function must use the [Autograd.t] parameter nodes it receives
    so the gradients computed by [Autograd.backward] reach those parameters. *)

type tensor = float Tensor.tensor

type forward = Autograd.t list -> tensor -> Autograd.t

type history = {
  train_losses : float list;
  (** Training MSE after each epoch's parameter update. *)
  validation_losses : float list option;
  (** Validation MSE after each epoch, if validation data was supplied. *)
}

val update_parameters : float -> Autograd.t list -> tensor list
(** Update each parameter using its stored gradient; return new tensors. *)

val train_step :
  learning_rate:float ->
  forward:forward ->
  parameters:tensor list ->
  inputs:tensor ->
  targets:tensor ->
  tensor list * float
(** Perform one full-batch gradient descent step. Returns updated parameters
    and the MSE calculated BEFORE the update. *)

val predict :
  forward:forward -> parameters:tensor list -> inputs:tensor -> tensor
(** Evaluate the forward function without performing a parameter update. *)

val evaluate :
  forward:forward ->
  parameters:tensor list ->
  inputs:tensor ->
  targets:tensor ->
  float
(** Return the mean squared error for the supplied data. *)

val train :
  ?validation:(tensor * tensor) ->
  learning_rate:float ->
  epochs:int ->
  forward:forward ->
  parameters:tensor list ->
  inputs:tensor ->
  targets:tensor ->
  unit ->
  tensor list * history
(** Perform [epochs] full-batch gradient descent steps and return the final
    parameters and post-update training/optional validation losses. *)
