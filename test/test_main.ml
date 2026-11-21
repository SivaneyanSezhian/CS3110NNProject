(* Run the real executable in isolated working directories so these tests do not
   depend on, or modify, the user's data/sample.csv. Dune passes the executable
   path as the first argument. *)
let executable =
  let path = Sys.argv.(1) in
  if Filename.is_relative path then Filename.concat (Sys.getcwd ()) path
  else path

type result = {
  status : Unix.process_status;
  stdout : string;
  stderr : string;
}

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let write_file path contents =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_string channel contents)

let rec remove_tree path =
  if Sys.is_directory path then begin
    Array.iter
      (fun name -> remove_tree (Filename.concat path name))
      (Sys.readdir path);
    Unix.rmdir path
  end
  else Sys.remove path

let with_fixture ?csv test =
  let directory = Filename.temp_file "test_main_" "" in
  Sys.remove directory;
  Unix.mkdir directory 0o700;
  let previous_directory = Sys.getcwd () in
  Fun.protect
    ~finally:(fun () ->
      Sys.chdir previous_directory;
      remove_tree directory)
    (fun () ->
      Sys.chdir directory;
      Unix.mkdir "data" 0o700;
      Option.iter (write_file "data/sample.csv") csv;
      test ())

let run arguments =
  let stdout = open_out_bin "stdout" in
  let stderr = open_out_bin "stderr" in
  Fun.protect
    ~finally:(fun () ->
      close_out_noerr stdout;
      close_out_noerr stderr)
    (fun () ->
      let pid =
        Unix.create_process executable
          (Array.of_list (executable :: arguments))
          Unix.stdin
          (Unix.descr_of_out_channel stdout)
          (Unix.descr_of_out_channel stderr)
      in
      let _, status = Unix.waitpid [] pid in
      { status; stdout = read_file "stdout"; stderr = read_file "stderr" })

let assert_exit expected result =
  if result.status <> Unix.WEXITED expected then
    let actual =
      match result.status with
      | Unix.WEXITED code -> Printf.sprintf "exit %d" code
      | Unix.WSIGNALED signal -> Printf.sprintf "signal %d" signal
      | Unix.WSTOPPED signal -> Printf.sprintf "stopped by signal %d" signal
    in
    failwith
      (Printf.sprintf "Expected exit %d, got %s\nstdout: %s\nstderr: %s"
         expected actual result.stdout result.stderr)

let assert_contains text fragment =
  let rec contains_from index =
    index + String.length fragment <= String.length text
    && (String.sub text index (String.length fragment) = fragment
       || contains_from (index + 1))
  in
  if not (contains_from 0) then
    failwith (Printf.sprintf "Expected %S in %S" fragment text)

let assert_error arguments message =
  let result = run arguments in
  assert_exit 2 result;
  assert (result.stdout = "");
  assert_contains result.stderr message

let test_help () =
  List.iter
    (fun option ->
      with_fixture (fun () ->
          let result = run [ option ] in
          assert_exit 0 result;
          assert (result.stderr = "");
          List.iter
            (assert_contains result.stdout)
            [ "Usage:"; "--learning-rate"; "--epochs"; "0.01"; "100" ]))
    [ "--help"; "-help" ]

let test_invalid_learning_rates () =
  List.iter
    (fun rate ->
      with_fixture (fun () ->
          assert_error
            [ "--learning-rate"; rate ]
            "learning rate must be positive and finite"))
    [ "0"; "-0.1"; "nan"; "inf"; "-inf" ]

let test_invalid_epochs () =
  List.iter
    (fun epochs ->
      with_fixture (fun () ->
          assert_error [ "--epochs"; epochs ] "epochs must be positive"))
    [ "0"; "-1" ]

let test_invalid_arguments () =
  List.iter
    (fun (arguments, message) ->
      with_fixture (fun () -> assert_error arguments message))
    [
      ([ "--learning-rate"; "abc" ], "expects a float");
      ([ "--epochs"; "1.5" ], "expects an integer");
      ([ "--learning-rate" ], "needs an argument");
      ([ "--epochs" ], "needs an argument");
      ([ "--unknown" ], "unknown option");
      ([ "unexpected.csv" ], "Unexpected argument: unexpected.csv");
    ]

let test_missing_csv () =
  with_fixture (fun () -> assert_error [ "--epochs"; "1" ] "data/sample.csv")

let test_nonnumeric_csv () =
  with_fixture ~csv:"1,2,3,4,5,6,7,8,9,10,label\n" (fun () ->
      assert_error [ "--epochs"; "1" ] "float_of_string")

let csv_rows count =
  let buffer = Buffer.create (count * 50) in
  for _ = 1 to count do
    Buffer.add_string buffer "0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8,0.9,-1,0.25\n"
  done;
  Buffer.contents buffer

let test_insufficient_rows () =
  List.iter
    (fun count ->
      with_fixture ~csv:(csv_rows count) (fun () ->
          assert_error [ "--epochs"; "1" ] "not enough elements in list"))
    [ 0; 8999 ]

let test_wrong_feature_count () =
  let csv = String.concat "" (List.init 9001 (fun _ -> "1,2,3\n")) in
  with_fixture ~csv (fun () ->
      assert_error [ "--epochs"; "1" ]
        "Mlp.forward: inputs must have shape [batch; input_dim]")

let test_training () =
  (* The fixed split needs 8000 training rows, 1000 validation rows, and at
     least one test row. One epoch keeps this end-to-end test bounded. *)
  with_fixture ~csv:(csv_rows 9001) (fun () ->
      let result = run [ "--epochs"; "1"; "--learning-rate"; "0.005" ] in
      assert_exit 0 result;
      assert (result.stderr = ""))

let () =
  List.iter
    (fun (name, test) ->
      try test ()
      with exn ->
        failwith (Printf.sprintf "%s: %s" name (Printexc.to_string exn)))
    [
      ("help without a dataset", test_help);
      ("invalid learning rates", test_invalid_learning_rates);
      ("invalid epochs", test_invalid_epochs);
      ("invalid arguments", test_invalid_arguments);
      ("missing CSV", test_missing_csv);
      ("nonnumeric CSV", test_nonnumeric_csv);
      ("insufficient rows", test_insufficient_rows);
      ("wrong feature count", test_wrong_feature_count);
      ("training", test_training);
    ];
  print_endline "Main executable tests passed"
