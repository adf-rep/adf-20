package utils.drift;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class ConceptDrift {
    public int p;
    public int width;
    public String concept;
    public String newConcept;

    ConceptDrift(int p, int width, String concept, String newConcept) {
        this.p = p;
        this.width = width;
        this.concept = concept;
        this.newConcept = newConcept;
    }
}
