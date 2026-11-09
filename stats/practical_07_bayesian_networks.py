"""Practical 7: Build and analyze a Bayesian network using pgmpy."""

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


def main() -> None:
    # Step 1: Define a directed graph: rain affects sprinklers and wet grass,
    # and sprinklers can also make the grass wet.
    model = DiscreteBayesianNetwork(
        [
            ("Rain", "Sprinkler"),
            ("Rain", "WetGrass"),
            ("Sprinkler", "WetGrass"),
        ]
    )

    # Step 2: Add the prior probability of rain.
    rain = TabularCPD(
        variable="Rain",
        variable_card=2,
        values=[[0.8], [0.2]],
        state_names={"Rain": ["No", "Yes"]},
    )

    # Step 3: Add the conditional probability of the sprinkler given rain.
    sprinkler = TabularCPD(
        variable="Sprinkler",
        variable_card=2,
        values=[[0.6, 0.99], [0.4, 0.01]],
        evidence=["Rain"],
        evidence_card=[2],
        state_names={
            "Sprinkler": ["Off", "On"],
            "Rain": ["No", "Yes"],
        },
    )

    # Step 4: Add the conditional probability of wet grass for each combination
    # of rain and sprinkler states.
    wet_grass = TabularCPD(
        variable="WetGrass",
        variable_card=2,
        values=[
            [0.99, 0.20, 0.10, 0.01],
            [0.01, 0.80, 0.90, 0.99],
        ],
        evidence=["Rain", "Sprinkler"],
        evidence_card=[2, 2],
        state_names={
            "WetGrass": ["No", "Yes"],
            "Rain": ["No", "Yes"],
            "Sprinkler": ["Off", "On"],
        },
    )

    # Step 5: Attach the probability tables and verify that they form a valid
    # Bayesian network.
    model.add_cpds(rain, sprinkler, wet_grass)
    if not model.check_model():
        raise ValueError("Bayesian network validation failed.")
    print("Bayesian network is valid.")
    print(f"Nodes: {list(model.nodes())}")
    print(f"Edges: {list(model.edges())}")

    # Step 6: Use exact inference to find the probability of rain after observing
    # wet grass.
    inference = VariableElimination(model)
    posterior = inference.query(
        variables=["Rain"],
        evidence={"WetGrass": "Yes"},
        show_progress=False,
    )
    print("\nProbability of rain given wet grass:")
    print(posterior)
    print(
        "P(Rain=Yes | WetGrass=Yes): "
        f"{posterior.get_value(Rain='Yes'):.4f}"
    )

    # Step 7: Also query the sprinkler state without any observations to view
    # its prior marginal distribution.
    sprinkler_marginal = inference.query(
        variables=["Sprinkler"],
        show_progress=False,
    )
    print("\nPrior probability of sprinkler state:")
    print(sprinkler_marginal)


if __name__ == "__main__":
    main()
