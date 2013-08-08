package com.fujitsu.ca.fic.dataloaders;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SequenceFileDatasetInfo {
    private static final Logger LOG = LoggerFactory.getLogger(SequenceFileDatasetInfo.class);

    public static int getFeatureCount(Configuration conf, String trainPath) throws IOException {
        try {
            int featureCount = 0;
            FileSystem fs = FileSystem.get(conf);
            try (SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(trainPath), conf)) {
                LongWritable key = new LongWritable();
                VectorWritable value = new VectorWritable();
                reader.next(key, value);
                NamedVector v = (NamedVector) value.get();
                featureCount = v.size();
            }
            return featureCount;
        } catch (IOException e) {
            LOG.error("Could not get the number of features from the provided sequence file. Please check path.");
            throw e;
        }
    }

}
