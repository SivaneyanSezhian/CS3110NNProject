open CS3110NNProject

let a = Tensor.Zero 0.
let () = print_endline (Tensor.string_of_tensor a)
let a' = Tensor.pointwise (fun x -> x +. x) a
let b = Tensor.make_tensor [ 2; 3; 2 ] (fun _ -> 0.)
let () = print_endline (Tensor.string_of_tensor b)
let () = Tensor.check_dims b
let b1 = Tensor.make_tensor [ 2; 3; 2 ] (fun _ -> 1.)
let b2 = Tensor.make_tensor [ 2; 3; 2 ] (fun _ -> 2.)
let b3 = Tensor.pointwise2 (fun x y -> x +. y) b1 b2
let () = print_endline (Tensor.string_of_tensor b3)
let () = Tensor.check_dims b3

let c =
  Tensor.make_tensor [ 3; 4; 5 ] (fun lst ->
      float_of_int (List.fold_left ( + ) 0 lst))

let () = print_endline (Tensor.string_of_tensor c)
let c' = Tensor.pointwisen (fun lst -> List.fold_left ( *. ) 1. lst) [ c; c; c ]
let () = print_endline (Tensor.string_of_tensor c')
let () = Tensor.check_dims c'

let d =
  Tensor.make_tensor [ 2; 3; 4 ] (fun lst ->
      float_of_string (String.concat "" (List.map string_of_int lst)))

let () = print_endline (Tensor.string_of_tensor d)
let d' = Tensor.flatten d
let () = print_endline (Tensor.string_of_tensor d')
let () = print_endline (string_of_float (Tensor.lookup d [ 1; 0; 1 ]))

let () =
  print_endline
    (string_of_float
       (Tensor.flattened_lookup d' (Tensor.get_dims d) [ 1; 0; 1 ]))

let d'' = Tensor.reshape d' [ 2; 3; 4 ]
let () = print_endline (Tensor.string_of_tensor d'')
let d''' = Tensor.reshape d'' [ 4; 3; 2 ]
let () = print_endline (Tensor.string_of_tensor d''')
let e1 = Tensor.tensor_of_2d_list [ [ 1; 1; 1 ]; [ 0; 1; 1 ]; [ -1; 0; 1 ] ]
let e1' = Tensor.pointwise float_of_int e1
let () = print_endline (Tensor.string_of_tensor e1')
let e2 = Tensor.tensor_of_2d_list [ [ 1; 1 ]; [ 0; 1 ]; [ 0; 1 ] ]
let e2' = Tensor.pointwise float_of_int e2
let () = print_endline (Tensor.string_of_tensor e2')
let e = Tensor.matmul e1' e2'
let () = print_endline (Tensor.string_of_tensor e)
