package com.fujitsu.ca.fic.classifiers.drivers;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fujitsu.ca.fic.classifiers.MahoutClassifierWrapper;
import com.google.common.collect.Lists;

/**
 * Launch a basic classification task using AdaptiveLogisticRegression and the DynamicDatasetLoader to train it one example at a time show
 * metrics of classification results on the console
 */
public class BasicMahoutClassificationDriver extends Configured implements Tool {
    private static Logger log = LoggerFactory.getLogger(BasicMahoutClassificationDriver.class);

    private static List<String> symbols = Lists.newArrayList("0", "1");

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BasicMahoutClassificationDriver(), args);
        System.exit(exitCode);
    }

    @Override
    public int run(String[] arg0) throws IOException {
        log.info("Classification of Sieve:Corpus6 with Mahout 0.8 with an online regression algorithm");
        int jobResult = Job.RUNNING;

        Configuration conf = getConf();
        String trainPath = conf.get("data.train.path");
        String testPath = conf.get("data.test.path");
        // String trainPath = "data/out/sieve/bns/spam-vs-rel/train-seq";
        // String testPath = "data/out/sieve/bns/spam-vs-rel/test-seq";

        if (trainPath == null | testPath == null) {
            log.error("The configuration file was not loaded correctly! Please check conf file is loaded: \n" + "data.train.path \n"
                    + "data.test.path \n");
            throw new IllegalStateException("The expected configuration values for data paths have not been found.");
        }
        try {
            MahoutClassifierWrapper mahoutWrapper = new MahoutClassifierWrapper(symbols);
            CrossFoldLearner bestLearner = mahoutWrapper.trainBestLearner(conf, trainPath);

            mahoutWrapper.test(conf, testPath, bestLearner);
            mahoutWrapper.showClassificationReport();

            // TODO output parameters to System.out
            // mahoutWrapper.showClassifierParameters(bestModelDyn);

            jobResult = Job.SUCCESS;
        } catch (IOException e) {
            log.error(e.toString());
            jobResult = Job.FAILED;
        }
        return jobResult;
    }
}
