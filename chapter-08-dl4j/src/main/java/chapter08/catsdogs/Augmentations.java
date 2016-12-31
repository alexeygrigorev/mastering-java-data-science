package chapter08.catsdogs;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

import org.apache.commons.io.FileUtils;
import org.imgscalr.Scalr;
import org.imgscalr.Scalr.Rotation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class Augmentations {

    private static final Logger LOGGER = LoggerFactory.getLogger(Augmentations.class);

    public static void main(String[] args) throws IOException {
        String root = "/home/datascience/data/catdogs";
        String sourceDirName = "train_cv";

        File sourceDir = new File(root, sourceDirName);
        File outputDir = new File(root, sourceDirName + "_simple");
        outputDir.mkdir();

        Iterator<File> files = FileUtils.iterateFiles(sourceDir, new String[] { "jpg" }, false);
        List<File> all = Lists.newArrayList(files);

        Random rnd = new Random(2);

        for (File f : all) {
            LOGGER.info("processing {}...", f);
            BufferedImage src = ImageIO.read(f);

            Rotation[] rotations = Rotation.values();

            for (Rotation rotation : rotations) {
                BufferedImage rotated = Scalr.rotate(src, rotation);
                String rotatedFile = f.getName() + "_" + rotation.name() + ".jpg";
                File outputFile = new File(outputDir, rotatedFile);
                ImageIO.write(rotated, "jpg", outputFile);

                int width = src.getWidth();
                int x = rnd.nextInt(width / 2);
                int w = (int) ((0.7 + rnd.nextDouble() / 2) * width / 2);

                int height = src.getHeight();
                int y = rnd.nextInt(height / 2);
                int h = (int) ((0.7 + rnd.nextDouble() / 2) * height / 2);

                if (x + w > width) {
                    w = width - x;
                }

                if (y + h > height) {
                    h = height - y;
                }

                BufferedImage crop = Scalr.crop(src, x, y, w, h);
                rotated = Scalr.rotate(crop, rotation);

                String cropppedFile = f.getName() + "_" + x + "_" + w + "_" + 
                            y + "_" + h + "_" + rotation.name() + ".jpg";
                outputFile = new File(outputDir, cropppedFile);
                ImageIO.write(rotated, "jpg", outputFile);
            }
        }

    }

}
