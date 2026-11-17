type 'a tensor =
  | Zero of 'a
  | Pos of 'a tensor_record

and 'a tensor_record = {
  dims : int list;
  ndarray : 'a tensor Array.t;
}

val check_dims : 'a tensor -> unit
(** [check_dims tnsr] checks if tensor [tnsr] matches the dimensions specified
    for itself and all of its sub-tensors. *)

val string_of_tensor : float tensor -> string
(** [string_of_tensor tnsr] returns a string representation of [tnsr]. *)

val make_tensor : int list -> (int list -> 'a) -> 'a tensor
(** [make_tensor dims f] makes a tensor of shape [dims] where each element has
    value [f i1 i2 ... in], here each [ik] is the index of the element in
    dimension [k]. *)

val zeroes : int list -> float tensor
(** [zeroes dims] makes a tensor of shape [dims] where each element is [0.]. *)

val ones : int list -> float tensor
(** [ones dims] makes a tensor of shape [dims] where each element is [1.]. *)

val random : int list -> float tensor
(** [random dims] makes a tensor of shape [dims] where each element is a random
    float from 0 to 1. *)

val same_dims : 'a tensor -> 'b tensor -> bool
(** [same_dims tnsr1 tnsr2] evaluates to [true] if and only if tensors [tnsr1]
    [tnsr2] have the same dimensions and [false] otherwise. *)

val pointwise : ('a -> 'b) -> 'a tensor -> 'b tensor
(** [pointwise f tnsr1] returns the tensor where the function [f] is applied to
    each element of [tnsr1] elementwise. *)

val pointwise2 : ('a -> 'b -> 'c) -> 'a tensor -> 'b tensor -> 'c tensor
(** [pointwise2 f tnsr1 tnsr2] returns the tensor where the function [f] is
    applied to each element of [tnsr1] and [tnsr2] elementwise. *)

val pointwisen : ('a list -> 'b) -> 'a tensor list -> 'b tensor
(** [pointwisen f tnsr_lst] returns the tensor where the function [f] is applied
    to each element of [tnsr_lst] elementwise. *)

val get_dims : 'a tensor -> int list
(** [get_dims tnsr] returns the list of dimensions of tensor [tnsr]. *)

val wrap : 'a tensor -> 'a tensor
(** [wrap tnsr] wraps [tnsr] in a one-dimensional tensor. *)

val wrap_k : 'a tensor -> int -> 'a tensor
(** [wrap_k tnsr k] wraps [tnsr] in a one-dimensional tensor [k] times. *)

val flatten : 'a tensor -> 'a tensor
(** [flatten tnsr] flattens into a one-dimensional tensor. *)

val lookup : 'a tensor -> int list -> 'a
(** [lookup tnsr indx_lst] looks up the element of [tnsr] with indices
    [indx_lst]. *)

val flattened_lookup : 'a tensor -> int list -> int list -> 'a
(** [flattened_lookup flat_tnsr tnsr_dims indx_lst] looks up the element of
    [flat_tnsr] treated as the flattened verion of a tensor with dimensions
    [tnsr_dims] with indices [indx_lst]. *)

val contract : float tensor -> float tensor -> int -> int -> float tensor
(** [contract tnsr1 tnsr2 indx1 indx2] contracts tensors [tnsr1] and [tnsr2] at
    indices [indx1] and [indx2]. *)

val matmul : float tensor -> float tensor -> float tensor
(** [matmul tnsr1 tnsr2] multiplies tensors [tnsr1] and [tnsr2] treated as
    matrices. *)

val tensor_of : 'a -> 'a tensor
(** [tensor_of x] returns [x] as a zero-dimensional tensor. *)

val tensor_of_1d_array : 'a array -> 'a tensor
(** [tensor_of_1d_array arr] returns [arr] as a one-dimensional tensor. *)

val tensor_of_1d_list : 'a list -> 'a tensor
(** [tensor_of_1d_list lst] returns [lst] as a one-dimensional tensor. *)

val stack_list : 'a tensor list -> 'a tensor
(** [stack_list tnsr_lst] returns the tensor that is the result of stacking the
    tensors in list [tnsr_lst]. *)

val stack_array : 'a tensor array -> 'a tensor
(** [stack_array tnsr_arr] returns the tensor that is the result of stacking the
    tensors in array [tnsr_arr]. *)

val tensor_of_2d_list : 'a list list -> 'a tensor
(** [tensor_of_2d_list lst_mtrx] returns [lst_mtrx] as a two-dimensional tensor.
*)

val tensor_of_2d_array : 'a array array -> 'a tensor
(** [tensor_of_2d_array arr_mtrx] returns [arr_mtrx] as a two-dimensional
    tensor. *)

val swap_dims : 'a tensor -> int -> int -> 'a tensor
(** [swap_dims tnsr i j] interchanges the [i]th and [j]th coordinates of [tnsr].
*)

val permute_dims : 'a tensor -> int list -> 'a tensor
(** [permute_dims tnsr prm] permutes the coordinates of [tnsr] with [prm]. *)

val group : 'a tensor -> int -> 'a tensor
(** [group tnsr k] arranges the sub-tensors of [tnsr] into groups of [k],
    treating such groups as a new dimension. *)

val group_dims : 'a tensor -> int list -> 'a tensor
(** [group_dims tnsr dims] recursively arranges the sub-tensors of [tnsr] into
    groups of [dims] such that the dimensions of the groupings are in order of
    [dims]. *)

val reshape : 'a tensor -> int list -> 'a tensor
(** [reshape tnsr dims] reshapes [tnsr] with dimensions [dims]. *)

val flatten_one : 'a tensor -> 'a tensor
(** [flatten_one tnsr] flattens one dimension of [tnsr]. *)

val flatten_k : 'a tensor -> int -> 'a tensor
(** [flatten_k tnsr k] flattens [k] dimensions of [tnsr]. *)

val sum : float tensor -> float
(** [sum x] returns the float sum of all elements in [x]. *)

val size : 'a tensor -> int
(** [size x] is the number of the elements in [x]. *)
