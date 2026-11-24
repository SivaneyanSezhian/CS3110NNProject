(** Reverse-mode automatic differentiation for graphs of float tensors.

    Wrap inputs and parameters with {!make}, build a scalar loss using the graph
    operations, then call {!backward} and read the parameter gradients with
    {!grad}. *)

type t
(** A tensor-valued node in a computation graph for reverse-mode automatic
    differentiation. *)

val make : float Tensor.tensor -> t
(** [make value] creates a leaf node from a well-formed tensor [value]. The node
    shares its storage, so treat [value] as read-only while using graphs built
    from the node. *)

val add : t -> t -> t
(** [add a b] returns a node holding the element-wise sum of [a] and [b]. The
    operands must have the same shape.

    @raise Assert_failure if the operand shapes differ. *)

val matmul : t -> t -> t
(** [matmul a b] returns a node holding the matrix product of [a] and [b]. Both
    operands must be rank-2 tensors, and the number of columns in [a] must equal
    the number of rows in [b].

    @raise Assert_failure
      if either operand has the wrong rank or the inner dimensions differ. *)

val sigmoid : t -> t
(** [sigmoid x] returns a node holding the element-wise sigmoid of [x]. *)

val relu : t -> t
(** [relu x] returns a node holding the element-wise ReLU of [x]. The backward
    pass uses a derivative of zero at zero. *)

val tanh : t -> t
(** [tanh x] returns a node holding the element-wise tanh of [x]. *)

val mse_loss : t -> float Tensor.tensor -> t
(** [mse_loss pred target] returns a scalar node holding the mean squared error
    over all elements. The target must be a well-formed tensor of the same shape
    as [pred], and both must contain at least one element. Backpropagation
    treats [target] as a constant and propagates the gradient through [pred].

    @raise Assert_failure if the prediction and target shapes differ. *)

val bce_loss : t -> float Tensor.tensor -> t
(** [bce_loss pred target] returns a scalar node holding the mean binary
    cross-entropy over all elements. Both tensors must be nonempty, have the
    same shape, and contain probabilities in [0, 1]. Targets are constants.
    Predictions are clipped to [1e-7, 1 - 1e-7] for numerical stability.
    Backpropagation uses zero derivative where predictions are clipped,
    including at either clipping boundary.

    @raise Assert_failure if the prediction and target shapes differ. *)

val backward : t -> unit
(** [backward loss] computes the gradients of the scalar [loss] with respect to
    each node it depends on. The call replaces the stored gradients for those
    nodes and sets the gradient of [loss] itself to one.

    @raise Assert_failure if [loss] is not scalar. *)

val value : t -> float Tensor.tensor
(** [value x] returns the tensor stored in [x] directly. Treat it as read-only
    while using graphs built from [x]. *)

val grad : t -> float Tensor.tensor
(** [grad x] returns the stored gradient with respect to [x] from the most
    recent call to {!backward} whose loss was [x] or depended on [x]. Before any
    such call, it returns zeros of the same shape as [x]. Treat the returned
    tensor as read-only. *)
