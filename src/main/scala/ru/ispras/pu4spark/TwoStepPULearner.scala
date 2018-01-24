package ru.ispras.pu4spark

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Performs PU learning in a 2-step manner:
  * on the first step, choose between all unlabeled examples those are negative with high probability
  * (so called, reliable negatives),
  * so that at the second step use them along with positive examples for training binary classifier.
  *
  * @author Nikita Astrakhantsev (astrakhantsev@ispras.ru)
  */
abstract class TwoStepPULearner[
  E <: ProbabilisticClassifier[Vector, E, M],
  M <: ProbabilisticClassificationModel[Vector, M]](
    classifier: ProbabilisticClassifier[Vector, E, M]) extends PositiveUnlabeledLearner {

  /**
    * Extracts probability instead of binary prediction
    */
  val getPOne = udf((v: Vector) => v(1))

  /**
    * Train binary classifier by considering all unlabeled data as negative data,
    * then apply it to all unlabeled data in order to have some measure of reliability of these negatives.
    *
    * @param df                 dataframe to work with
    * @param labelColumnName    name for column containing positive or unlabeled label
    * @param featuresColumnName name for column containing features as a vector (e.g. after VectorAssembler)
    * @param finalLabel         name for column that will contain required measure of reliability of these negatives
    * @return updated datafram with finalLabel column
    */
  def zeroStep(df: DataFrame, labelColumnName: String, featuresColumnName: String, finalLabel: String): DataFrame = {

    val labelIndexer = new StringIndexer()
      .setInputCol(labelColumnName)
      .setOutputCol(ProbabilisticClassifierConfig.labelName)

    //scaler seems to not improve results
    //StandardScaler with mean scaling requires DenseVectors, while VectorAssembler can return only SparseVectors
    //    val scaler = new MinMaxScaler().setInputCol(srcFeaturesName).setOutputCol(scaledFeaturesName)

    // RF requires that (from Spark example, 'Automatically identify categorical features, and index them')
    val featureIndexer = new VectorIndexer()
      .setInputCol(featuresColumnName)
      .setOutputCol(ProbabilisticClassifierConfig.featuresName)
      .setMaxCategories(4)  //features with > 4 distinct values are treated as continuous.

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer))
    val preparedDf = pipeline.fit(df).transform(df)

    val model: M = classifier.fit(preparedDf)
    val predictions: DataFrame = model.transform(preparedDf)
    predictions.withColumn(finalLabel, getPOne(predictions("probability")))
  }
}
