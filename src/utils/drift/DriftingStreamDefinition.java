package utils.drift;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class DriftingStreamDefinition {
    public String streamName;
    public String originStreamName;
    public ArrayList<ConceptDrift> drifts;
    public int streamSize;
    public int logGranularity;

    private DriftingStreamDefinition(String streamName, String originStreamName, ArrayList<ConceptDrift> drifts, int streamSize, int logGranularity) {
        this.streamName = streamName;
        this.originStreamName = originStreamName;
        this.drifts = drifts;
        this.streamSize = streamSize;
        this.logGranularity = logGranularity;
    }

    public static ArrayList<DriftingStreamDefinition> createDriftDefinitions() {
        ArrayList<DriftingStreamDefinition> definitions = new  ArrayList<>();

        definitions.add(new DriftingStreamDefinition(
                "SEA_1",
                "SEA",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(150000, 100, "SEA_concept_2", "SEA_concept_3"),
                        new ConceptDrift(300000, 100, "SEA_concept_3", "SEA_concept_4"),
                        new ConceptDrift(450000, 100, "SEA_concept_4", "SEA_concept_3")
                )),
                600000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "SEA_2",
                "SEA",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(150000, 10000, "SEA_concept_2", "SEA_concept_3"),
                        new ConceptDrift(300000, 10000, "SEA_concept_3", "SEA_concept_4"),
                        new ConceptDrift(450000, 10000, "SEA_concept_4", "SEA_concept_3")
                )),
                600000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "STAGGER_1",
                "STAGGER",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(150000, 100, "STAGGER_concept_1", "STAGGER_concept_2"),
                        new ConceptDrift(300000, 100, "STAGGER_concept_2", "STAGGER_concept_3"),
                        new ConceptDrift(450000, 100, "STAGGER_concept_3", "STAGGER_concept_1")
                )),
                600000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "STAGGER_2",
                "STAGGER",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(150000, 10000, "STAGGER_concept_1", "STAGGER_concept_2"),
                        new ConceptDrift(300000, 10000, "STAGGER_concept_2", "STAGGER_concept_3"),
                        new ConceptDrift(450000, 10000, "STAGGER_concept_3", "STAGGER_concept_1")
                )),
                600000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "RBF_1",
                "RBF",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(250000, 100, "RBF_concept_1_n5", "RBF_concept_2_n5"),
                        new ConceptDrift(500000, 100, "RBF_concept_2_n5", "RBF_concept_4_n5"),
                        new ConceptDrift(750000, 100, "RBF_concept_4_n5", "RBF_concept_5_n5")
                )),
                1000000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "RBF_2",
                "RBF",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(250000, 10000, "RBF_concept_1_n5", "RBF_concept_2_n5"),
                        new ConceptDrift(500000, 10000, "RBF_concept_2_n5", "RBF_concept_4_n5"),
                        new ConceptDrift(750000, 10000, "RBF_concept_4_n5", "RBF_concept_5_n5")
                )),
                1000000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "RBF_3",
                "RBF",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(400000, 50000, "RBF_concept_11_n15_12m", "RBF_concept_12_n15_12m"),
                        new ConceptDrift(800000, 50000, "RBF_concept_12_n15_12m", "RBF_concept_13_n15_12m")
                )),
                1200000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "RBF_4",
                "RBF",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(400000, 100000, "RBF_concept_11_n15_12m", "RBF_concept_12_n15_12m"),
                        new ConceptDrift(800000, 100000, "RBF_concept_12_n15_12m", "RBF_concept_13_n15_12m")
                )),
                1200000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "TREE_1",
                "TREE",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(250000, 100, "TREE_concept_2", "TREE_concept_1"),
                        new ConceptDrift(500000, 100, "TREE_concept_1", "TREE_concept_3"),
                        new ConceptDrift(750000, 100, "TREE_concept_3", "TREE_concept_4")
                )),
                1000000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "TREE_2",
                "TREE",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(250000, 10000, "TREE_concept_2", "TREE_concept_1"),
                        new ConceptDrift(500000, 10000, "TREE_concept_1", "TREE_concept_3"),
                        new ConceptDrift(750000, 10000, "TREE_concept_3", "TREE_concept_4")
                )),
                1000000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "TREE_3",
                "TREE",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(400000, 50000, "TREE_concept_2_12m", "TREE_concept_1_12m"),
                        new ConceptDrift(800000, 50000, "TREE_concept_1_12m", "TREE_concept_3_12m")
                )),
                1200000,
                100)
        );

        definitions.add(new DriftingStreamDefinition(
                "TREE_4",
                "TREE",
                new ArrayList<>(Arrays.asList(
                        new ConceptDrift(400000, 100000, "TREE_concept_2_12m", "TREE_concept_1_12m"),
                        new ConceptDrift(800000, 100000, "TREE_concept_1_12m", "TREE_concept_3_12m")
                )),
                1200000,
                100)
        );

        return definitions;
    }

    public static HashMap<String, DriftingStreamDefinition> createDriftDefinitionsMap() {
        ArrayList<DriftingStreamDefinition> definitions = DriftingStreamDefinition.createDriftDefinitions();
        HashMap<String, DriftingStreamDefinition> definitionsMap = new HashMap<>();

        for (DriftingStreamDefinition definition : definitions) {
            definitionsMap.put(definition.streamName, definition);
        }

        return definitionsMap;
    }
}
