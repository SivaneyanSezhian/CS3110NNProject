type unary_function = {
  f_1 : float -> float;
  grad_1 : float -> float;
}

type binary_function = {
  f_2 : float * float -> float;
  grad_2 : float * float -> float * float;
}

type nary_function = {
  f_n : float list -> float;
  grad_n : float list -> float list;
}

type primitive_function =
  | Unary of unary_function
  | Binary of binary_function
  | Nary of nary_function

type nfunction =
  | Const of float
  | Var of string
  | Function of primitive_function * nfunction list

val eval : nfunction -> (string * float) list -> float
(** [eval nfunc assignments] evaluates [nfunc] with the assignments specified in
    [assignments]. Raises [Not_found] if a variable included is not given an
    assignment. *)

val grad_layer : nfunction -> (string * float) list -> (string * float) list
(** [grad_layer nfunc assignments] computes the gradient of [nfunc] with respect
    to each variable in [assignments] evaluated at the point corresponding to
    the values specified in [assignments]. *)

val id : unary_function
val id_nary : nary_function
val add : binary_function
val sub : binary_function
val mult : binary_function
val div : binary_function
val exp : unary_function
val sum_f : float list -> float
val grad_f : 'a list -> float list
val sum_n : nary_function
val sig_f_1 : float -> float
val sig_grad_1 : float -> float
val sig_f : unary_function
val relu_f : unary_function
val tanh_f : unary_function
