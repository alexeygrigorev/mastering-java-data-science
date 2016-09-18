package chapter04.classification;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import chapter04.RankedPageData;
import chapter04.cv.Dataset;
import chapter04.cv.Split;
import chapter04.preprocess.StandardizationPreprocessor;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.SolverType;

public class PagePredictionLibLinear {

    public static void main(String[] args) throws IOException {
        Split split = RankedPageData.readRankedPagesMatrix();

        Dataset train = split.getTrain();
        Dataset test = split.getTest();

        StandardizationPreprocessor preprocessor = StandardizationPreprocessor.train(train);
        train = preprocessor.transform(train);
        test = preprocessor.transform(test);

        LibLinear.mute();

        List<Split> folds = train.kfold(3);

        SolverType[] solvers = { SolverType.L1R_LR, SolverType.L2R_LR, SolverType.L1R_L2LOSS_SVC,
                SolverType.L2R_L2LOSS_SVC };
        double[] Cs = { 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0 };

        for (SolverType solver : solvers) {
            for (double C : Cs) {
                DescriptiveStatistics summary = LibLinear.crossValidate(folds, fold -> {
                    Parameter param = new Parameter(solver, C, 0.0001);
                    return LibLinear.train(fold, param);
                });

                double mean = summary.getMean();
                double std = summary.getStandardDeviation();
                System.out.printf("%15s  C=%7.3f, auc=%.4f ± %.4f%n", solver, C, mean, std);
            }
        }

        Parameter param = new Parameter(SolverType.L1R_LR, 0.05, 0.0001);
        Model finalModel = LibLinear.train(train, param);
        double finalAuc = LibLinear.auc(finalModel, test);
        System.out.printf("final logreg        auc=%.4f%n", finalAuc);
    }

}
