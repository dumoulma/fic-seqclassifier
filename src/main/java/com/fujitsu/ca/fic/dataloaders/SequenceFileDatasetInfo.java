package com.fujitsu.ca.fic.dataloaders;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class SequenceFileDatasetInfo {
    private static Logger log = LoggerFactory.getLogger(SequenceFileDatasetInfo.class);

    private SequenceFileDatasetInfo() {
    }

    public static int getFeatureCount(Configuration conf, String trainPath) throws IOException {
        try {
            int featureCount = 0;
            FileSystem fs = FileSystem.get(conf);
            SequenceFile.Reader reader = null;
            try {
                reader = new SequenceFile.Reader(fs, new Path(trainPath + "/part-r-00000"), conf);

                Text key = new Text();
                VectorWritable value = new VectorWritable();
                reader.next(key, value);
                NamedVector v = (NamedVector) value.get();
                featureCount = v.size();
            } finally {
                if (reader != null) {
                    reader.close();
                }
            }
            return featureCount;
        } catch (IOException e) {
            log.error("Could not get the number of features from the provided sequence file. Please check path.");
            throw e;
        }
    }
}
