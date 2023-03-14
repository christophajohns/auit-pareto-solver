"""Functions to create computational user models implementing various utility functions."""

from typing import List
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

def create_random_runtimes(seed: int) -> pd.DataFrame:
    """
    Generate a random pd.DataFrame with the required format for the `plot_runtimes` function.

    Parameters
    ----------
    seed : int
        Seed for the random number generator. Used to ensure reproducibility of the generated DataFrame.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns corresponding to the features of an optimization run and runtime.

    Notes
    -----
    The function uses numpy's default_rng to ensure better performance and reproducibility of the random numbers.
    """
    rng = np.random.default_rng(seed)

    # Set up the dataframe
    results = []
    n_runs_per_config = 10
    run_id = 0
    for scenario in ["CONV+LIN", "NCONV+NLIN"]:
        for solver in ["WS", "Ours"]:
                for n_proposals in [1, 10]:
                    if solver == "Ours" and n_proposals == 1:
                        continue
                    for run_iter in range(n_runs_per_config):
                        run_id += 1
                        seed = rng.integers(0, 1000)
                        start_time = dt.datetime.now() + dt.timedelta(seconds=rng.integers(1, 60).item())
                        end_time = start_time + dt.timedelta(seconds=rng.integers(1, 60).item())
                        runtime = (end_time - start_time).total_seconds()
                        results.append({
                            "run_id": run_id,
                            "scenario": scenario,
                            "solver": solver,
                            "n_proposals": n_proposals,
                            "run_iter": run_iter+1,
                            "seed": seed,
                            "start_time": start_time,
                            "end_time": end_time,
                            "runtime": runtime
                        })
    df = pd.DataFrame(results)
    return df

def plot_runtimes_for_scenario(ax: plt.Axes, scenario: str, solvers: List[str], n_proposals: List[int], solver_labels: List[str], runtimes: pd.DataFrame) -> None:
    """
    Plots boxplots of the runtimes for each configuration and solver for a given scenario.
    
    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        The axes object to plot on.
    scenario: str
        The scenario to plot for.
    solvers: List[str]
        The solvers to include in the plot.
    n_proposals: List[int]
        The number of proposals to include in the plot.
    solver_labels: List[str]
        The labels for the solvers to include in the plot.
    runtimes: pandas.DataFrame
        DataFrame containing the runtime information for each configuration, solver and number of proposals.
        
    Returns:
    --------
    None
    """
    # Get the data for the current scenario
    scenario_data = runtimes[runtimes["scenario"] == scenario]

    # For each solver (WS and Ours)...
    for j, solver in enumerate(solvers):

        # Get the data for the current solver
        solver_data = scenario_data[scenario_data["solver"] == solver]

        # For each number of proposals (1 and 10)...
        for k, n_prop in enumerate(n_proposals):

            if solver == "Ours" and n_prop == 1:
                continue

            # Get the data for the current number of proposals
            n_prop_data = solver_data[solver_data["n_proposals"] == n_prop]

            # Get the runtimes for the current solver and number of proposals
            config_runtimes = n_prop_data["runtime"].values

            # Create a boxplot for the current solver and number of proposals
            ax.boxplot(config_runtimes, positions=[solver_labels.index(f"{solver} ({n_prop} prop)")],
                        patch_artist=True, widths=0.5, showfliers=True, showmeans=True, meanline=True,
                        boxprops={"facecolor": "lightgrey"})

    # Set labels for the x-axis ticks
    ax.set_xticks(np.arange(len(solver_labels)))
    ax.set_xticklabels(solver_labels)

def plot_runtimes(runtimes: pd.DataFrame) -> plt.Figure:
    """
    Creates a boxplot figure showing the runtime performance of different solvers for two optimization scenarios. 
    Scenario 1 (CONV+LIN) assumes end-user preferences can be expressed as a linear combination of optimization objectives, 
    forming a convex Pareto frontier. Scenario 2 (NCONV+NLIN) assumes end-user preferences cannot be expressed as a 
    linear combination of the optimization objectives, forming a non-convex Pareto frontier.

    The boxplot has the runtime on the y-axis and the solver and number of proposals on the x-axis. There are two boxplots 
    next to each other, one for each scenario. The weighted sum solver further has a boxplot for each number of proposals, 
    and the boxplots for the weighted sum solver are close together and separated from the other solvers. The y-axis is 
    labeled with "Runtime (s)" and the x-axis is labeled with "Optimizer (No. Proposals)". The x-axis is stacked with the 
    number of proposals at the top and the solver name at the bottom. The order of the boxplots is WS (1 prop), WS (10 prop), 
    Ours. The boxplots include whiskers and outliers, and the median is marked with a line. The boxplots are colored in a 
    way that is easy to distinguish between the different solvers.

    Parameters:
    -----------
    runtimes : pandas.DataFrame
        DataFrame containing the runtime data to be plotted. It should have the following columns: 
        ["run_id", "scenario", "solver", "n_proposals", "run_iter", "seed", "start_time", "end_time", "runtime"]

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plotted boxplots.
    """
    # Define colors for the solvers
    colors = {"WS (1 prop)": "tab:blue", "WS (10 prop)": "tab:green", "Ours": "tab:orange"}

    # Create figure and subplots
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Set figure title
    # fig.suptitle("Runtime Performance")

    # Set titles for subplots
    axs[0].set_title("CONV+LIN")
    axs[1].set_title("NCONV+NLIN")

    # Define labels for x-axis and y-axis
    axs[0].set_ylabel("Runtime (s)")
    axs[1].set_ylabel("Runtime (s)")
    axs[0].set_xlabel("Optimizer (No. Proposals)")
    axs[1].set_xlabel("Optimizer (No. Proposals)")

    # Define the limits for the y-axis
    axs[0].set_ylim(0, 60)
    axs[1].set_ylim(0, 60)

    # Define the solvers and number of proposals
    scenarios = ["CONV+LIN", "NCONV+NLIN"]
    solver_labels = ["WS (1 prop)", "WS (10 prop)", "Ours (10 prop)"]
    solvers = ["WS", "WS", "Ours"]
    n_proposals = [1, 10, 10]


    # For each scenario (CONV+LIN and NCONV+NLIN)...
    for i, scenario in enumerate(scenarios):

        plot_runtimes_for_scenario(axs[i], scenario, solvers, n_proposals, solver_labels, runtimes)

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Show the plot
    # plt.show()

    # Return the figure object
    return fig

    
def create_random_utilities(seed: int) -> pd.DataFrame:
    """
    Generate a random pd.DataFrame with the required format for the `plot_max_utilities` function.

    Parameters
    ----------
    seed : int
        Seed for the random number generator. Used to ensure reproducibility of the generated DataFrame.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns corresponding to the features of an optimization run and max utility.

    Notes
    -----
    The function uses numpy's default_rng to ensure better performance and reproducibility of the random numbers.
    """
    rng = np.random.default_rng(seed)

    # Set up the dataframe
    results = []

    n_runs_per_config = 10
    n_utility_functions = 100
    run_id = 0
    adaptation_id = 0
    
    for scenario in ["CONV+LIN", "NCONV+NLIN"]:
        for solver in ["WS", "Ours"]:
                for n_proposals in [1, 10]:
                    for run_iter in range(n_runs_per_config):
                        run_id += 1
                        for utility_id in range(1, n_utility_functions+1):
                            for adaptation in range(n_proposals):
                                adaptation_id += 1
                                utility = rng.uniform()
                                results.append({
                                    "run_id": run_id,
                                    "utility_id": utility_id,
                                    "adaptation_id": adaptation_id,
                                    "utility": utility,
                                })

    df = pd.DataFrame(results)
    return df

def get_max_utilities(runtimes: pd.DataFrame, utilities: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas DataFrame containing the maximum utility for each optimization run.

    Parameters
    ----------
    runtimes : pd.DataFrame
        A pandas DataFrame with the run information for the optimization runs.
        Must contain columns "run_id", "scenario", "solver", "n_proposals".
    utilities : pd.DataFrame
        A pandas DataFrame with the utility information for each optimization run.
        Must contain columns "run_id", "utility_id", "utility".

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with columns corresponding to the scenario, solver, number of proposals,
        run ID, and maximum utility for each utility type.

    Notes
    -----
    The function merges the runtimes and utilities dataframes and computes the maximum utility for each
    utility type and optimization run.

    The resulting DataFrame has one row per utility type and optimization run, and the columns
    correspond to the scenario, solver, number of proposals, run ID, and maximum utility.
    """
    # Add the run information to the utilities DataFrame
    utilities = utilities.merge(runtimes[["run_id", "scenario", "solver", "n_proposals"]], on="run_id")

    # For each scenario, solver, n_proposals and run_id, get the max utility per utility_id
    max_utilities = utilities.groupby(["scenario", "solver", "n_proposals", "run_id", "utility_id"])["utility"].max().reset_index()

    # Rename the utility column to max_utility
    max_utilities = max_utilities.rename(columns={"utility": "max_utility"})

    # Return the max utilities
    return max_utilities

def plot_max_utilities_for_scenario(ax: plt.Axes, scenario: str, solvers: List[str], n_proposals: List[int], solver_labels: List[str], max_utilities: pd.DataFrame, exp_utility: float) -> None:
    """
    Plots boxplots of the max utilities for each configuration and solver for a given scenario.
    
    Parameters:
    -----------
    ax: matplotlib.axes.Axes
        The axes object to plot on.
    scenario: str
        The scenario to plot for.
    solvers: List[str]
        The solvers to include in the plot.
    n_proposals: List[int]
        The number of proposals to include in the plot.
    solver_labels: List[str]
        The labels for the solvers to include in the plot.
    max_utilities: pandas.DataFrame
        DataFrame containing the max utility information for each configuration, solver and number of proposals.
    exp_utility: float
        The expected utility for the scenario.
        
    Returns:
    --------
    None
    """
    # Get the data for the current scenario
    scenario_data = max_utilities[max_utilities["scenario"] == scenario]

    # For each solver (WS and Ours)...
    for j, solver in enumerate(solvers):

        # Get the data for the current solver
        solver_data = scenario_data[scenario_data["solver"] == solver]

        # For each number of proposals (1 and 10)...
        for k, n_prop in enumerate(n_proposals):

            if solver == "Ours" and n_prop == 1:
                continue

            # Get the data for the current number of proposals
            n_prop_data = solver_data[solver_data["n_proposals"] == n_prop]

            # Get the max utilities for the current solver and number of proposals
            config_utilities = n_prop_data["max_utility"].values

            # Create a boxplot for the current solver and number of proposals
            ax.boxplot(config_utilities, positions=[solver_labels.index(f"{solver} ({n_prop} prop)")],
                        patch_artist=True, widths=0.5, showfliers=True, showmeans=True, meanline=True, boxprops={"facecolor": "lightgrey"})

    # Set labels for the x-axis ticks
    ax.set_xticks(np.arange(len(solver_labels)))
    ax.set_xticklabels(solver_labels)

    # Add a dashed horizontal line for the expected utility
    ax.axhline(y=exp_utility, color="black", linestyle="--", alpha=0.2, zorder=0, label="Exp. Utility", linewidth=1)

def plot_max_utilities(runtimes: pd.DataFrame, utilities: pd.DataFrame, expected_utilities: dict) -> plt.Figure:
    """
    Plots boxplots of the max utilities for each configuration and scenario.
    
    Parameters:
    -----------
    runtimes: pandas.DataFrame
        DataFrame containing the runtime information for each run.
    utilities: pandas.DataFrame
        DataFrame containing the utility information for each run.
    expected_utilities: dict
        Dictionary containing the expected utility for each scenario.
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plotted boxplots.
    """
    # Create figure and subplots
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Set figure title
    # fig.suptitle("Max Utility Performance")

    # Set titles for subplots
    axs[0].set_title("CONV+LIN")
    axs[1].set_title("NCONV+NLIN")

    # Define labels for x-axis and y-axis
    axs[0].set_ylabel("Max Utility")
    axs[1].set_ylabel("Max Utility")
    axs[0].set_xlabel("Optimizer (No. Proposals)")
    axs[1].set_xlabel("Optimizer (No. Proposals)")

    # Define the limits for the y-axis
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)

    # Get a DataFrame with the max utilities for each condition and configuration
    max_utilities = get_max_utilities(runtimes, utilities)

    # Determine the scenarios, solvers and number of proposals
    scenarios = ["CONV+LIN", "NCONV+NLIN"]
    solvers = ["WS", "Ours"]
    n_proposals = [1, 10]
    solver_labels = ["WS (1 prop)", "WS (10 prop)", "Ours (10 prop)"]

    # For each scenario (CONV+LIN and NCONV+NLIN)...
    for i, scenario in enumerate(scenarios):

        plot_max_utilities_for_scenario(axs[i], scenario, solvers, n_proposals, solver_labels, max_utilities, expected_utilities[scenario])

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Show the plot
    # plt.show()

    # Return the figure object
    return fig

# Display the max utility and runtime performance in one figure
def plot_results(runtimes: pd.DataFrame, utilities: pd.DataFrame, expected_utilities: dict) -> plt.Figure:
    """
    Plots the runtime and max utility performance in one figure where each scenario is one column and the
    utility/runtime performance are rows.

    Parameters:
    -----------
    runtimes: pandas.DataFrame
        DataFrame containing runtime data for each run.

    utilities: pandas.DataFrame
        DataFrame containing utility data for each run.

    expected_utilities: dict
        Dictionary containing the expected utility for each scenario.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing the plotted boxplots.
    """
    # Create figure and subplots
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    # Set plot titles
    axs[0][0].set_title("CONV+LIN")
    axs[0][1].set_title("NCONV+NLIN")
    # axs[1][0].set_title("CONV+LIN")
    # axs[1][1].set_title("NCONV+NLIN")

    # Set labels for the y-axis
    axs[0][0].set_ylabel("Max Utility")
    # axs[0][1].set_ylabel("Max Utility")
    axs[1][0].set_ylabel("Runtime (s)")
    # axs[1][1].set_ylabel("Runtime (s)")

    # Set labels for the x-axis
    axs[1][0].set_xlabel("Optimizer (No. Proposals)")
    axs[1][1].set_xlabel("Optimizer (No. Proposals)")

    # Get a DataFrame with the max utilities for each condition and configuration
    max_utilities = get_max_utilities(runtimes, utilities)

    # Determine the scenarios, solvers and number of proposals
    scenarios = ["CONV+LIN", "NCONV+NLIN"]
    solvers = ["WS", "Ours"]
    n_proposals = [1, 10]
    solver_labels = ["WS (1 prop)", "WS (10 prop)", "Ours (10 prop)"]

    # For each scenario (CONV+LIN and NCONV+NLIN)...
    for i, scenario in enumerate(scenarios):

        # Plot the max utility performance
        plot_max_utilities_for_scenario(axs[0][i], scenario, solvers, n_proposals, solver_labels, max_utilities, expected_utilities[scenario])

        # Plot the runtime performance
        plot_runtimes_for_scenario(axs[1][i], scenario, solvers, n_proposals, solver_labels, runtimes)

    # Adjust the layout of the subplots
    fig.tight_layout()

    # Show the plot
    # plt.show()

    # Return the figure object
    return fig

# Create a DataFrame view with summary statistics for the max utilities and runtimes for each configuration
def get_results_df(runtimes: pd.DataFrame, utilities: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a DataFrame view with summary statistics for the max utilities and runtimes for each configuration.

    Parameters:
    -----------
    runtimes: pandas.DataFrame
        DataFrame containing runtime data for each run.

    utilities: pandas.DataFrame
        DataFrame containing utility data for each run.

    Returns:
    --------
    pandas.DataFrame
        DataFrame view with summary statistics for the max utilities and runtimes for each configuration.
    """
    # Get a DataFrame with the max utilities for each condition and configuration
    max_utilities = get_max_utilities(runtimes, utilities)

    # Add runtime data to the DataFrame
    max_utilities = max_utilities.merge(runtimes[["run_id", "runtime"]], on="run_id")

    # Group the DataFrame by scenario, solver and number of proposals
    grouped = max_utilities.groupby(["scenario", "solver", "n_proposals"])

    # Get the mean and standard deviation for the max utilities and runtimes
    results = grouped.agg({"max_utility": ["mean", "std"], "runtime": ["mean", "std"]})

    # Rename the columns
    results.columns = ["mean_max_utility", "std_max_utility", "mean_runtime", "std_runtime"]

    # Reset the index
    results = results.reset_index()

    # Return the DataFrame
    return results