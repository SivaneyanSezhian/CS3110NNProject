(** Multi-layer perceptrons built from {!Autograd} operations.

    Describe a network with {!create}, get starting weights with
    {!init_parameters}, then pass [forward net] and those weights to
    {!Training.train}. Inputs have shape [[batch; input_dim]], one example per
    row. Parameters come in one weight and bias pair per layer, first layer
    first. *)

(** An element-wise function applied to the output of a layer. *)
type activation =
  | Identity
  | Relu
  | Sigmoid
  | Tanh

type layer = {
  units : int;
  activation : activation;
}
(** A fully connected layer with [units] outputs. *)

type t
(** An MLP architecture. It holds no weights. *)

val create : input_dim:int -> layer list -> t
(** [create ~input_dim layers] returns an architecture that reads [input_dim]
    features and applies [layers] in order. The last layer is the output layer.

    @raise Invalid_argument
      if [input_dim] or any layer size is not positive, or [layers] is empty. *)

val input_dim : t -> int
(** [input_dim net] returns the number of features per example. *)

val output_dim : t -> int
(** [output_dim net] returns the number of units in the output layer. *)

val layers : t -> layer list
(** [layers net] returns the layers of [net] in order. *)

val num_parameters : t -> int
(** [num_parameters net] returns the total number of weights and biases. *)

val init_parameters : ?seed:int -> t -> float Tensor.tensor list
(** [init_parameters ?seed net] returns fresh parameters for [net]. Each weight
    matrix has shape [[inputs; units]] and is drawn around zero, scaled by the
    layer's input width. Each bias has shape [[1; units]] and starts at zero.
    The same [seed] gives the same parameters. *)

val forward : t -> Autograd.t list -> float Tensor.tensor -> Autograd.t
(** [forward net params inputs] returns a node holding the output of [net] on
    [inputs]. Each layer computes [activation (X W + b)], adding [b] to every
    row. [forward net] has type {!Training.forward}.

    @raise Invalid_argument
      if [params] does not match the shapes from {!init_parameters}, or [inputs]
      is not of shape [[batch; input_dim net]]. *)

val activation_of_string : string -> activation option
(** [activation_of_string s] returns the activation named by [s], ignoring case
    and surrounding whitespace. Accepted names are ["identity"], ["linear"],
    ["relu"], ["sigmoid"] and ["tanh"]. *)

val string_of_activation : activation -> string
(** [string_of_activation a] returns the lowercase name of [a]. *)

val to_string : t -> string
(** [to_string net] returns a summary of each layer and the parameter count. *)
