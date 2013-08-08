package com.fujitsu.ca.fic.classifiers.metrics;

import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.math.stats.OnlineAuc;

/*
 * NOTE: This is a value class, it is not explicitely tested
 */
public class ClassificationMetrics {
    private ConfusionMatrix cm = null;
    private OnlineAuc auc = null;
    private final int tp;
    private final int tn;
    private final int fp;
    private final int fn;

    public ClassificationMetrics(ConfusionMatrix cm, OnlineAuc auc) {
        this.cm = cm;
        this.auc = auc;

        tn = cm.getConfusionMatrix()[0][0];
        tp = cm.getConfusionMatrix()[1][1];
        fn = cm.getConfusionMatrix()[0][1];
        fp = cm.getConfusionMatrix()[1][0];
    }

    public double precision() {
        return (double) tp / (tp + fn);
    }

    public double recall() {
        return (double) tp / (tp + fp);
    }

    public double trueNegativeRate() {
        return (double) tn / (tn + fp);
    }

    public double accuracy() {
        return (double) (tp + tn) / (tp + tn + fp + fn);
    }

    public ConfusionMatrix confusionMatrix() {
        return cm;
    }

    public double negativePrecision() {
        return (double) tn / (tn + fp);
    }

    public double negativeRecall() {
        return (double) tn / (tn + fn);
    }

    public double auc() {
        return auc.auc();
    }

    public void showReport() {
        System.out.println("-------------------------------------------");
        System.out.println("Classification report:");
        System.out.println("-------------------------------------------");
        System.out.printf("AUC               : %.4f\n", auc.auc());
        System.out.printf("Positive precision: %.2f\n", precision() * 100);
        System.out.printf("Positive recall   : %.2f\n", recall() * 100);
        System.out.printf("Negative precision: %.2f\n", precision() * 100);
        System.out.printf("Negative recall   : %.2f\n", recall() * 100);
        System.out.printf("True negative rate: %.2f\n", trueNegativeRate() * 100);
        System.out.printf("Accuracy          : %.2f\n", accuracy() * 100);
        System.out.println();
    }

}
