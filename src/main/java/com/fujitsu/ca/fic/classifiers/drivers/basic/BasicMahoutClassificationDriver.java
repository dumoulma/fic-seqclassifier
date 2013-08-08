package com.fujitsu.ca.fic.classifiers.drivers.basic;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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
    private static final Logger LOG = LoggerFactory.getLogger(BasicMahoutClassificationDriver.class);

    private static List<String> SYMBOLS = Lists.newArrayList("0", "1");

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new BasicMahoutClassificationDriver(), args);
        System.exit(exitCode);
    }

    @Override
    public int run(String[] arg0) throws Exception {
        LOG.info("Classification of Sieve:Corpus6 with Mahout 0.8 with an online regression algorithm");
        int jobResult = Job.RUNNING;

        Configuration conf = getConf();
        if (conf.get("data.test.path") == null) {
            loadConfigurationFile(conf);
        }

        try {
            MahoutClassifierWrapper mahoutWrapper = new MahoutClassifierWrapper(SYMBOLS);
            String trainPath = conf.get("data.train.path");

            CrossFoldLearner bestLearner = mahoutWrapper.trainBestLearner(conf, trainPath);

            String testPath = conf.get("data.test.path");
            mahoutWrapper.test(conf, testPath, bestLearner);
            mahoutWrapper.showClassificationReport();

            // TODO output parameters to System.out
            // mahoutWrapper.showClassifierParameters(bestModelDyn);

            jobResult = Job.SUCCESS;
        } catch (IOException e) {
            LOG.error(e.toString());
            jobResult = Job.FAILED;
        }
        return jobResult;
    }

    private void loadConfigurationFile(Configuration conf) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        conf.addResource(fs.open(new Path("conf/local-conf.xml")));
    }

}
