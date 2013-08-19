package com.fujitsu.ca.fic.classifiers;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ConfusionMatrix;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.GlobalOnlineAuc;
import org.apache.mahout.math.stats.OnlineAuc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fujitsu.ca.fic.classifiers.metrics.ClassificationMetrics;
import com.fujitsu.ca.fic.dataloaders.SequenceFileDatasetInfo;

public class MahoutClassifierWrapper {
    private static final Logger LOG = LoggerFactory.getLogger(MahoutClassifierWrapper.class);

    private final int categories;
    private final List<String> symbols;
    private final String defaultValue = "unknown";
    private ClassificationMetrics metrics = null;

    public MahoutClassifierWrapper(List<String> symbols) {
        this.symbols = symbols;
        categories = symbols.size();
    }

    public void train(OnlineLearner onlineLearner, List<NamedVector> trainset) {
        LOG.info("Training OnlineLearner using list of vectors");
        try {
            for (NamedVector example : trainset) {
                String labelName = example.getName();
                int actual = Integer.parseInt(labelName);
                onlineLearner.train(actual, example);
            }
        } finally {
            onlineLearner.close();
        }
        LOG.info("Training complete!");
    }

    public void train(OnlineLearner onlineLearner, Iterable<NamedVector> vectorIterable) {
        LOG.info("Training OnlineLearner using dynamic dataloader");
        try {
            for (NamedVector nextExample : vectorIterable) {
                String docLabel = nextExample.getName();
                String labelIndex = null;
                try {
                    labelIndex = docLabel.split(",")[1];
                } catch (NumberFormatException nfe) {
                    LOG.warn("Couldn't parse label of this document: " + docLabel);
                    continue;
                }
                onlineLearner.train(Integer.parseInt(labelIndex), nextExample);
            }
        } finally {
            onlineLearner.close();
        }
        LOG.info("Training complete!");
    }

    public void train(OnlineLearner onlineLearner, Iterable<NamedVector> vectorIterable, int trainSetSizeLimit) {
        LOG.info("Training OnlineLearner using dynamic dataloader with max examples=" + trainSetSizeLimit);
        try {
            int nRead = 0;
            for (NamedVector nextExample : vectorIterable) {
                String docLabel = nextExample.getName();
                String labelIndex = null;
                try {
                    labelIndex = docLabel.split(",")[1];
                } catch (NumberFormatException nfe) {
                    LOG.warn("Couldn't parse label of this document: " + docLabel);
                    continue;
                }
                onlineLearner.train(Integer.parseInt(labelIndex), nextExample);
                if (nRead++ == trainSetSizeLimit)
                    break;
            }
        } finally {
            onlineLearner.close();
        }
        LOG.info("Training complete!");
    }

    public CrossFoldLearner trainBestLearner(Configuration conf, String trainPath) throws IOException {
        AdaptiveLogisticRegression onlineLearner = new AdaptiveLogisticRegression(symbols.size(), SequenceFileDatasetInfo.getFeatureCount(
                conf, trainPath), new L1());
        SequenceFile.Reader reader = null;
        try {
            reader = new SequenceFile.Reader(FileSystem.get(conf), new Path(trainPath + "/part-r-00000"), conf);
            Text key = new Text();
            VectorWritable value = new VectorWritable();
            while (reader.next(key, value)) {
                NamedVector nextExample = (NamedVector) value.get();
                String label = nextExample.getName();
                int labelValue = -1;
                try {
                    labelValue = Integer.parseInt(label);
                } catch (NumberFormatException nfe) {
                    LOG.warn("Couldn't parse label of this document: " + key.toString());
                    continue;
                }
                onlineLearner.train(labelValue, nextExample);
            }
        } finally {
            if (reader != null) {
                reader.close();
            }
        }
        onlineLearner.close();
        CrossFoldLearner bestLearner = onlineLearner.getBest().getPayload().getLearner();
        return bestLearner;
    }

    public void test(Configuration conf, String testDirName, AbstractVectorClassifier classifier) throws IOException {
        LOG.info("Training classifier with test data using dynamic loader");

        ResultAnalyzer analyzer = new ResultAnalyzer(symbols, defaultValue);
        ConfusionMatrix cm = new ConfusionMatrix(symbols, defaultValue);
        OnlineAuc auc = new GlobalOnlineAuc();
        SequenceFile.Reader reader = null;
        try {
            reader = new SequenceFile.Reader(FileSystem.get(conf), new Path(testDirName + "/part-r-00000"), conf);

            Text key = new Text();
            VectorWritable value = new VectorWritable();
            while (reader.next(key, value)) {
                NamedVector nextExample = (NamedVector) value.get();
                String correctLabel = nextExample.getName();
                int actual = Integer.parseInt(correctLabel);

                Vector p = new DenseVector(categories);
                classifier.classifyFull(p, nextExample);
                int estimated = p.maxValueIndex();

                String estimatedLabel = String.valueOf(estimated);
                cm.addInstance(correctLabel, estimatedLabel);

                auc.addSample(actual, classifier.classifyScalar(nextExample));
                analyzer.addInstance(correctLabel, new ClassifierResult(estimatedLabel));

                // int bump = bumps[(int) Math.floor(step) % bumps.length];
                // int scale = (int) Math.pow(10, Math.floor(step / bumps.length));
                // if (i % (bump * scale) == 0) {
                // step += 0.25;
                // System.out.printf("%5d  %10.2f     %s %s\n", i, averageCorrect * 100, correctLabel, estimatedLabel);
                // }
                // ++i;

            }
            metrics = new ClassificationMetrics(cm, auc);
        } finally {
            if (reader != null) {
                reader.close();
            }
        }

        LOG.info("Testing complete!");
    }

    public void showClassificationReport() {
        if (metrics == null) {
            LOG.warn("Can't show classification report on untrained learner. Please train learner first!");
            throw new RuntimeException("Must do test() before showClassificationReport()");
        }

        metrics.showReport();
        System.out.printf("%s\n\n", metrics.confusionMatrix().toString());
    }

    public ClassificationMetrics getMetrics() {
        return metrics;
    }

    public void showClassifierParameters(CrossFoldLearner bestModelDyn) {
        throw new UnsupportedOperationException("Not implemented yet!");
    }
}
