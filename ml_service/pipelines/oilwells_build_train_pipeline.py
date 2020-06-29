from azureml.core import Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment


def main():
    e = Env()
    print(e.__dict__)
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name, subscription_id=e.subscription_id, resource_group=e.resource_group
    )
    print("get_workspace:")
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
    if aml_compute is not None:
        print("aml_compute:")
        print(aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(aml_workspace, e.aml_env_name, create_new=e.rebuild_env)  #
    run_config = RunConfiguration()
    run_config.environment = environment

    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables["DATASTORE_NAME"] = datastore_name  # NOQA: E501

    model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)
    dataset_version_param = PipelineParameter(name="dataset_version", default_value=e.dataset_version)
    data_file_path_param = PipelineParameter(name="data_file_path", default_value="none")
    caller_run_id_param = PipelineParameter(name="caller_run_id", default_value="none")

    # Get dataset name
    dataset_name = e.dataset_name

    # Check to see if dataset exists
    if dataset_name not in aml_workspace.datasets:
        raise ValueError(f"can't find dataset {dataset_name} in datastore {datastore_name}")

    # Create PipelineData to pass data between steps
    model_data = PipelineData("model_data", datastore=aml_workspace.get_default_datastore())
    train_ds = (
        PipelineData("train_ds", datastore=aml_workspace.get_default_datastore())
        .as_dataset()
        .parse_delimited_files()
        .register(name="train", create_new_version=True)
    )
    test_ds = (
        PipelineData("test_ds", datastore=aml_workspace.get_default_datastore())
        .as_dataset()
        .parse_delimited_files()
        .register(name="test", create_new_version=True)
    )

    prepare_step = PythonScriptStep(
        name="Prepare Data",
        script_name=e.prepare_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        outputs=[train_ds, test_ds],
        arguments=[
            "--dataset_version",
            dataset_version_param,
            "--data_file_path",
            data_file_path_param,
            "--dataset_name",
            dataset_name,
            "--caller_run_id",
            caller_run_id_param,
            "--train_ds",
            train_ds,
            "--test_ds",
            test_ds
        ],
        runconfig=run_config,
        allow_reuse=True,
    )
    print("Step Prepare created")

    train_step = PythonScriptStep(
        name="Train Model",
        script_name=e.train_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        inputs=[train_ds.as_named_input("training_data"), test_ds.as_named_input("testing_data")],
        outputs=[model_data],
        arguments=["--model_name", model_name_param, "--model_data", model_data],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Train created")

    evaluate_step = PythonScriptStep(
        name="Evaluate Model ",
        script_name=e.evaluate_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        arguments=["--model_name", model_name_param, "--allow_run_cancel", e.allow_run_cancel,],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    register_step = PythonScriptStep(
        name="Register Model ",
        script_name=e.register_script_path,
        compute_target=aml_compute,
        source_directory=e.sources_directory_train,
        inputs=[model_data],
        arguments=["--model_name", model_name_param, "--step_input", model_data],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Register created")
    # Check run_evaluation flag to include or exclude evaluation step.
    if (e.run_evaluation).lower() == "true":
        print("Include evaluation step before register step.")
        evaluate_step.run_after(train_step)
        register_step.run_after(evaluate_step)
        steps = [prepare_step, train_step, evaluate_step, register_step]
    else:
        print("Exclude evaluation step and directly run register step.")
        register_step.run_after(train_step)
        steps = [prepare_step, train_step, register_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=e.pipeline_name, description="Model training/retraining pipeline", version=e.build_id
    )
    print(f"Published pipeline: {published_pipeline.name}")
    print(f"for build {published_pipeline.version}")


if __name__ == "__main__":
    main()
