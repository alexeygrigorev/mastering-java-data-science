package chapter08.catsdogs;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;

import com.google.common.collect.Lists;

public class TrainValSplit {

    public static void main(String[] args) throws IOException {
        String root = "/home/agrigorev/tmp/data/cats-dogs";
        File trainDir = new File(root,  "train");
        double valFrac = 0.2;
        long seed = 1;

        Iterator<File> files = FileUtils.iterateFiles(trainDir, new String[] { "jpg" }, false);
        List<File> all = Lists.newArrayList(files);

        Random random = new Random(seed);
        Collections.shuffle(all, random);

        int trainSize = (int) (all.size() * (1 - valFrac));
        List<File> train = all.subList(0, trainSize);
        copyTo(train, new File(root, "train_cv"));

        List<File> val = all.subList(trainSize, all.size());
        copyTo(val, new File(root, "val_cv"));
    }

    private static void copyTo(List<File> pics, File dir) throws IOException {
        dir.mkdirs();

        for (File pic : pics) {
            FileUtils.copyFileToDirectory(pic, dir);
        }
    }
}
