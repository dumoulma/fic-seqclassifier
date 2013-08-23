package com.fujitsu.ca.fic.dataloaders;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.equalTo;

import static org.hamcrest.MatcherAssert.assertThat;

public class SequenceFileDatasetInfoTest {
    @Test
    public void test() {
        assertThat(new Integer(1), equalTo(1));
    }
}
