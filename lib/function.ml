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

let get_first lst =
  match lst with
  | h :: t -> h
  | _ -> raise (failwith "No first element")

let get_second lst =
  match lst with
  | h1 :: h2 :: t -> h2
  | _ -> raise (failwith "No second element")

let eval_prim prim_f args =
  match prim_f with
  | Unary f -> f.f_1 (get_first args)
  | Binary f -> f.f_2 (get_first args, get_second args)
  | Nary f -> f.f_n args

let rec eval nfunc assignments =
  match nfunc with
  | Const x -> x
  | Var v -> List.assoc v assignments
  | Function (prim_f, args) ->
      eval_prim prim_f
        (List.rev
           (List.fold_left (fun acc nf -> eval nf assignments :: acc) [] args))

let list_grad prim_f =
  match prim_f with
  | Unary f -> fun args -> [ f.grad_1 (get_first args) ]
  | Binary f ->
      fun args ->
        let g1, g2 = f.grad_2 (get_first args, get_second args) in
        [ g1; g2 ]
  | Nary f -> fun args -> f.grad_n args

let mult_assoc factor v = List.map (fun (x, a) -> (x, factor *. a)) v

let assoc_or_zero v assoc =
  match List.assoc_opt v assoc with
  | None -> 0.
  | Some x -> x

let add_assoc assoc1 assoc2 =
  let assoc1' =
    List.map
      (fun (v, x) -> (v, assoc_or_zero v assoc1 +. assoc_or_zero v assoc2))
      assoc1
  in
  let assoc2' =
    List.map
      (fun (v, x) -> (v, assoc_or_zero v assoc1 +. assoc_or_zero v assoc2))
      assoc2
  in
  List.sort_uniq Stdlib.compare (assoc1' @ assoc2')

let grad_prim prim_f args grad_lst =
  let df = list_grad prim_f args in
  let deltas =
    List.fold_left2 (fun acc d v -> mult_assoc d v :: acc) [] df grad_lst
  in
  List.fold_left (fun acc delta -> add_assoc acc delta) [] deltas

let rec grad_layer_rec nfunc assignments =
  match nfunc with
  | Const x -> (x, [])
  | Var v -> (List.assoc v assignments, [ (v, 1.) ])
  | Function (prim_f, args_f) ->
      let arg_grad_lst =
        List.map (fun nf -> grad_layer_rec nf assignments) args_f
      in
      let args = List.map fst arg_grad_lst in
      let grad_lst = List.map snd arg_grad_lst in
      (eval_prim prim_f args, grad_prim prim_f args grad_lst)

let id = { f_1 = (fun x -> x); grad_1 = (fun _ -> 1.) }
let id_nary = { f_n = get_first; grad_n = (fun _ -> [ 1. ]) }
let add = { f_2 = (fun (x, y) -> x +. y); grad_2 = (fun _ -> (1., 1.)) }
let sub = { f_2 = (fun (x, y) -> x -. y); grad_2 = (fun _ -> (1., -1.)) }
let mult = { f_2 = (fun (x, y) -> x *. y); grad_2 = (fun (x, y) -> (y, x)) }

let div =
  {
    f_2 = (fun (x, y) -> x /. y);
    grad_2 = (fun (x, y) -> (1. /. y, -1. *. x /. (y *. y)));
  }

let exp = { f_1 = (fun x -> exp x); grad_1 = (fun x -> x) }
let sum_f = List.fold_left (fun acc x -> acc +. x) 0.
let grad_f lst = List.fold_left (fun acc x -> 1. :: acc) [] lst
let sum_n = { f_n = sum_f; grad_n = grad_f }
