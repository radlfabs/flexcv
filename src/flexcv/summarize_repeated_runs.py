import neptune
import pandas as pd


def get_repeated_cv_metadata(str_children="Instance of repeated run ", api_dict=None):
    """This function can be used to fetch metadata from repeated cross-validation runs.
    We use it to get the ids of the children runs and their descriptions.

    Args:
        str_children (str): The string that is prepended to the description of each child run.
        api_dict (dict): A dictionary containing the Neptune.ai project name and the api token.
    """
    if api_dict is None:
        raise ValueError("api_dict must be provided")

    # get a list of all runs in the project
    project = neptune.init_project(
        project=api_dict["project"],
        api_token=api_dict["api_token"],
        mode="read-only",
    )
    runs_table_df = project.fetch_runs_table().to_pandas()
    # use only rows where "sys/description" begins with "Instance"
    # group by run sys/description
    grouped = runs_table_df[
        runs_table_df["sys/description"].str.startswith(str_children)
    ].groupby("sys/description")
    # get sys/id for each group
    grouped_ids = grouped["sys/id"].apply(list)
    # remove "Instance of repeated run " and trailing dot from the description
    grouped_ids.index = grouped_ids.index.str.replace(str_children, "")
    grouped_ids.index = grouped_ids.index.str.replace(".", "")
    # rename the index to "host id"
    grouped_ids.index.name = "host id"
    # rename the column to "children ids"
    grouped_ids.name = "children ids"
    metadata = pd.DataFrame(grouped_ids)
    # use the host ids to get their sys/description and make them a new column in the DataFrame
    host_ids = grouped_ids.index
    descriptions = runs_table_df[runs_table_df["sys/id"].isin(host_ids)][
        "sys/description"
    ]
    descriptions.index = host_ids
    descriptions.index.name = "host id"
    descriptions.name = "description"
    # join the two DataFrames
    metadata = metadata.join(pd.DataFrame(descriptions))
    # save to excel
    metadata.to_excel("repeated_cv_metadata.xlsx")


if __name__ == "__main__":
    api_dict = {
        "api_token": "ANONYMOUS",
        "project": "MYPROJECT",
    }
    str_children = "Instance of repeated run "
    get_repeated_cv_metadata(str_children=str_children, api_dict=api_dict)
