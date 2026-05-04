# Hybrid-Data-Driven-ML-DRO-PV-BESS-Microgrid-Optimizer

This project is created in Fulfillment of the Requirements for the Degree of B.Tech CSE-DS, AKTU


PV and BESS must be optimally sized to ensure reliable, low-cost power under uncertainty. This study proposes a hybrid data-driven and Distributionally Robust Optimization approach that jointly optimizes system capacity using real data, improving cost efficiency and reliability over traditional methods.

Photovoltaic (PV) systems and Battery Energy Storage Systems (BESS) must be optimally
sized to ensure flexible power delivery at minimum cost, especially under conditions of high
uncertainty. Rapid growth in renewable energy adoption has exposed critical gaps in conventional planning tools, particularly in regions where historical energy data is sparse or
unreliable. This study introduces a hybrid framework that integrates data-driven Machine
Learning (ML) with Distributionally Robust Optimization (DRO) to address the challenges
of optimal PV-BESS sizing in uncertain and data-scarce environments.
Traditional deterministic methods often fail to capture the volatility of renewable generation and load demand, while standard stochastic models require extensive historical datasets
that are rarely available in rural or remote regions. The proposed methodology utilizes ML
techniques, including regression-based forecasting and clustering algorithms, to predict demand and supply patterns, which directly inform the construction of ambiguity sets for the
DRO model. This approach allows for a “distributionally strong” optimization that protects
against worst-case scenarios and distributional shifts without requiring a fixed probability
distribution.
The framework is validated through the column-and-constraint generation (CnCG) algorithm, which efficiently solves the resulting robust optimization problem. Experimental
results demonstrate that the hybrid framework achieves a 15–20% reduction in total lifecycle costs compared to traditional deterministic approaches. Furthermore, the system shows
a 10–15% increase in the robustness margin, significantly reducing the likelihood of blackouts. By reducing energy loss by up to 50% and improving overall system reliability to 92%,
this framework provides a cost-effective and resilient solution for sustainable electrification
in data-scarce and off-grid regions.
