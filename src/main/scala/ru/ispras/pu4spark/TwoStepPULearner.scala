package ru.ispras.pu4spark

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf, when}
import org.apache.spark.sql.types.DoubleType

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
    * @return updated dataframe with finalLabel column
    */
  def zeroStep(df: DataFrame, labelColumnName: String, featuresColumnName: String, finalLabel: String): DataFrame = {
    val dfWithMeta = indexLabelColumn(df, labelColumnName, ProbabilisticClassifierConfig.labelName, Seq("0", "1"))

    //scaler seems to not improve results
    //StandardScaler with mean scaling requires DenseVectors, while VectorAssembler can return only SparseVectors
    //    val scaler = new MinMaxScaler().setInputCol(srcFeaturesName).setOutputCol(scaledFeaturesName)

    // RF requires that (from Spark example, 'Automatically identify categorical features, and index them')
    val featureIndexer = new VectorIndexer()
      .setInputCol(featuresColumnName)
      .setOutputCol(ProbabilisticClassifierConfig.featuresName)
      .setMaxCategories(4)  //features with > 4 distinct values are treated as continuous.

    val pipeline = new Pipeline().setStages(Array(featureIndexer))
    val preparedDf = pipeline.fit(dfWithMeta).transform(dfWithMeta)

    val model: M = classifier.fit(preparedDf)
    val predictions: DataFrame = model.transform(preparedDf)
    val res = predictions.withColumn(finalLabel, getPOne(predictions("probability")))
    res
  }

  /**
    * Adds meta-information to label column and casts it to DoubleType, so that it can be used for training.
    * StringIndexer can't be used, because it assigns index based on labels frequency.
    *
    *
    * @param df        dataframe to index label
    * @param inputCol  name of column with original label
    * @param outputCol name of column with indexed label
    * @param values    labels to support
    * @return dataframe with indexed label
    */
  def indexLabelColumn(df: DataFrame, inputCol: String, outputCol: String, values: Seq[String]): DataFrame = {
    val meta = NominalAttribute
      .defaultAttr
      .withName(inputCol)
      .withValues(values.head, values.tail: _*)
      .toMetadata

    df.withColumn(outputCol, col(inputCol).as(outputCol, meta).cast(DoubleType))
  }

  /**
    * Replaces one value in column by another and renames this column.
    * It is used to change labels from zero to special value indicating undefined.
    *
    * @param df            dataframe to replace value
    * @param origColName   name of column with original label
    * @param newColName    name of column with replaced label
    * @param value2replace value from that column that should be used instead of existing value
    *                      (i.e. if the value differs from value2keep, than it would be replaced by value2replace
    * @param value2keep    value that should be kept
    * @return dataframe with replaced values
    */
  def replaceZerosByUndefLabel(df: DataFrame,
                               origColName: String,
                               newColName: String,
                               value2replace: Double,
                               value2keep: Double = 1): DataFrame = {
    df.withColumn(newColName,
      when(col(origColName).equalTo(value2keep), value2keep).otherwise(value2replace))
      .drop(origColName)
  }
}
