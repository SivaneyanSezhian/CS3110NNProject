module Function = CS3110NNProject.Function

let assoc_equal assoc1 assoc2 =
  List.sort_uniq Stdlib.compare (assoc1 @ assoc2)
  = List.sort_uniq Stdlib.compare (assoc2 @ assoc1)

let exp_1 =
  Function.Function
    ( Function.Unary Function.exp,
      [ Function.Var "x"; Function.Var "y"; Function.Var "z" ] )

let sum_3 =
  Function.Function
    ( Function.Nary Function.sum_n,
      [
        Function.Function (Function.Unary Function.id, [ Function.Var "x" ]);
        Function.Function (Function.Unary Function.id, [ Function.Var "x" ]);
        Function.Function (Function.Unary Function.id, [ Function.Var "y" ]);
        Function.Function (Function.Unary Function.id, [ Function.Var "z" ]);
      ] )

let eval_test () =
  assert (Function.eval sum_3 [ ("x", 1.); ("y", -2.); ("z", 3.) ] = 3.);
  assert (Function.eval exp_1 [ ("x", 0.); ("y", -2.); ("z", 3.) ] = 1.)

let grad_test () =
  assert (
    assoc_equal
      (Function.grad_layer sum_3
         [ ("x", 1.); ("y", -2.); ("z", 3.); ("a", 4.) ])
      [ ("x", 2.); ("y", 1.); ("z", 1.) ]);
  assert (
    assoc_equal
      (Function.grad_layer exp_1
         [ ("x", 1.); ("y", -2.); ("z", 3.); ("a", 4.) ])
      [ ("x", 1.); ("y", 0.); ("z", 0.) ])

let () =
  List.iter
    (fun (name, test) ->
      try test ()
      with exn ->
        failwith (Printf.sprintf "%s: %s" name (Printexc.to_string exn)))
    [ ("eval_test", eval_test); ("grad_test", grad_test) ];
  print_endline "Function module tests passed"
