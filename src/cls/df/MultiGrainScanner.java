package cls.df;

import com.yahoo.labs.samoa.instances.*;
import utils.InstanceUtils;

import java.util.ArrayList;

class MultiGrainScanner {

    private int scanDepth;
    private int stride;
    private boolean image;

    MultiGrainScanner(int scanDepth, int stride, boolean image) {
        this.scanDepth = scanDepth;
        this.stride = stride;
        this.image = image;
    }

    ArrayList<Instances> scanInstance(Instance instance) {
        if (!this.image) return this.scan1D(instance);
        return this.scan2D(instance);
    }

    private ArrayList<Instances> scan1D(Instance instance) {
        ArrayList<Instances> multiGrainInstances = new ArrayList<>();

        for (int i = this.scanDepth; i > 0; i--) {
            multiGrainInstances.add(this.scan1DForDepth(i, instance));
        }

        return multiGrainInstances;
    }

    private Instances scan1DForDepth(int depth, Instance instance) {
        Instances instances = null;
        int N = instance.numAttributes() - 1;
        int n = (int)(N / Math.pow(2.0, depth));

        int s = this.stride > 0 ? this.stride : n;
        int up = s == 1 ? N - n + 1: N;

        for (int i = 0; i < up; i += s) {
            Attribute[] attributes = new Attribute[n + 1];
            double[] attValues = new double[n + 1];

            for (int j = 0; j < n; j++) {
                int idx = i + j; // (root pos) + (inter-window pos)
                if (idx < N) {
                    attributes[j] = instance.attribute(idx);
                    attValues[j] = instance.value(idx);
                } else {
                    attributes[j] = new Attribute("pad_" + j);
                    attValues[j] = 0.0;
                }
            }

            attributes[n] = instance.classAttribute();
            attValues[n] = instance.classValue();
            Instance subInstance = new InstanceImpl(1.0, attValues);
            InstancesHeader header = new InstancesHeader();
            header.setAttributes(attributes);
            header.setClassIndex(n);
            subInstance.setDataset(header);

            if (instances == null) instances = new Instances("", attributes, 0);
            instances.add(subInstance);
        }

        return instances;
    }

    private ArrayList<Instances> scan2D(Instance instance) {
        ArrayList<Instances> multiGrainInstances = new ArrayList<>();

        for (int i = this.scanDepth; i > 0; i--) {
            multiGrainInstances.add(this.scan2DForDepth(i, instance));
        }

        return multiGrainInstances;
    }

    private Instances scan2DForDepth(int depth, Instance instance) {
        Instances instances = null;
        int N = (int)Math.sqrt(instance.numAttributes() - 1);
        int n = (int)(N / Math.pow(2.0, depth));

        int s = this.stride > 0 ? this.stride : n;
        int up = s == 1 ? N - n + 1: N;

        for (int i = 0; i < up; i += s) {
            for (int j = 0; j < up; j += s) {
                Attribute[] attributes = new Attribute[n * n + 1];
                double[] attValues = new double[n * n + 1];

                for (int k = 0; k < n; k++) {
                    for (int m = 0; m < n; m++) {
                        int idx = (i * N + j) + k * N + m; // (root pos) + (inter-matrix pos)
                        if (i + k < N && j + m < N) {
                            attributes[k * n + m] = instance.attribute(idx);
                            attValues[k * n + m] = instance.value(idx);
                        } else {
                            attributes[k * n + m] = new Attribute("pad_" + idx);
                            attValues[k * n + m] = 0.0;
                        }
                    }
                }

                attributes[n * n] = instance.classAttribute();
                attValues[n * n] = instance.classValue();

                Instance subInstance = new InstanceImpl(1.0, attValues);
                InstancesHeader header = new InstancesHeader();
                header.setAttributes(attributes);
                header.setClassIndex(n * n);
                subInstance.setDataset(header);

                if (instances == null) instances = new Instances("", attributes, 0);
                instances.add(subInstance);
            }
        }

        return instances;
    }

}
