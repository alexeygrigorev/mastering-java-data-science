package chapter10.search;

import java.util.ArrayList;
import java.util.List;

import chapter07.searchengine.FeatureExtractor;
import chapter07.searchengine.PrepareData.QueryDocumentPair;
import joinery.DataFrame;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;

public class XgbRanker implements Ranker {

    private static final String NAME = "xgb";

    private final FeatureExtractor featureExtractor;
    private final Booster booster;

    public XgbRanker(FeatureExtractor featureExtractor, String pathToModel) throws Exception {
        this.featureExtractor = featureExtractor;
        this.booster = XGBoost.loadModel(pathToModel);
    }

    @Override
    public SearchResults rank(List<QueryDocumentPair> inputList) throws Exception {
        DataFrame<Double> featuresDf = featureExtractor.transform(inputList);
        double[][] matrix = featuresDf.toModelMatrix(0.0);

        double[] probs = XgbUtils.predict(booster, matrix);

        List<ScoredIndex> scored = ScoredIndex.wrap(probs);
        List<QueryDocumentPair> result = new ArrayList<>(inputList.size());
        for (ScoredIndex idx : scored) {
            QueryDocumentPair doc = inputList.get(idx.getIndex());
            result.add(doc);
        }

        return SearchResults.wrap(NAME, result);
    }

    @Override
    public String name() {
        return NAME;
    }

}
