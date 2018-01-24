package ru.ispras.pu4spark

import org.apache.logging.log4j.LogManager
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegressionModel, ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Original Positive-Unlabeled learning algorithm; firstly proposed in
  *  Liu, B., Dai, Y., Li, X. L., Lee, W. S., & Philip, Y. (2002).
  *  Partially supervised classification of text documents.
  *  In ICML 2002, Proceedings of the nineteenth international conference on machine learning. (pp. 387–394). <br>
  *
  * Pseudocode was taken from:
  *  Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015).
  *  Detecting positive and negative deceptive opinions using PU-learning.
  *  Information Processing & Management, 51(4), 433-443.
  *
  * @author Nikita Astrakhantsev (astrakhantsev@ispras.ru)
  */
class TraditionalPULearner[
    E <: ProbabilisticClassifier[Vector, E, M],
    M <: ProbabilisticClassificationModel[Vector, M]](
        relNegThreshold: Double,
        maxIters: Int,
        classifier: ProbabilisticClassifier[Vector, E, M]) extends TwoStepPULearner[E,M](classifier) {
  val log = LogManager.getLogger(getClass)

  override def weight(df: DataFrame, labelColumnName: String, featuresColumnName: String, finalLabel: String): DataFrame = {
    val oneStepPUDF: DataFrame = zeroStep(df, labelColumnName, featuresColumnName, finalLabel)
      .drop("probability").drop("prediction").drop("rawPrediction").drop(ProbabilisticClassifierConfig.labelName)

    val confAdder = new RelNegConfidenceThresholdAdder(relNegThreshold)

    val prevLabel = "prevLabel"
    val curLabel = "curLabel"
    var curDF = oneStepPUDF.withColumnRenamed(labelColumnName, prevLabel)

    for (i <- 1 to maxIters) {
      //replace weights by 0-1 column for further learning (induce lables for curLabDF)
      val curLabelColumn = confAdder.binarizeUDF(curDF(finalLabel), curDF(prevLabel))

      curDF = curDF.withColumn(curLabel, curLabelColumn).cache()
      val newRelNegCount = curDF
        //unlabeled in previous iterations && negative in current iteration
        .filter(curDF(prevLabel) === TraditionalPULearner.undefLabel && curDF(curLabel) === TraditionalPULearner.relNegLabel)
        .count()

      log.debug(s"newRelNegCount: $newRelNegCount")

      if (newRelNegCount == 0) {
        return curDF
      }

      //learn new classifier
      val curLabDF = curDF.filter(curDF(curLabel) !== TraditionalPULearner.undefLabel) //keep positives and rel. negs
      //      curLabDF.describe().show()
      val newLabelIndexer = new StringIndexer()
          .setInputCol(curLabel)
          .setOutputCol(ProbabilisticClassifierConfig.labelName)

      val newPipeline = new Pipeline().setStages(Array(newLabelIndexer)) //, classifier))
      val newPreparedDf = newPipeline.fit(curLabDF).transform(curLabDF)

      val model = classifier.fit(newPreparedDf)

//      log.debug(s"Coefficients: ${model.asInstanceOf[LogisticRegressionModel].coefficients} " +
//        s"Intercept: ${model.asInstanceOf[LogisticRegressionModel].intercept}")

      //apply classifier to still unlabeled data
      val labUnlabDF = model.transform(curDF)
      curDF = labUnlabDF.withColumn(finalLabel, getPOne(labUnlabDF("probability")))
        .drop("probability").drop("prediction").drop("rawPrediction").drop(ProbabilisticClassifierConfig.labelName)
      curDF = curDF.drop(prevLabel)
        .withColumnRenamed(curLabel, prevLabel)
    }
    curDF
  }
}

private class RelNegConfidenceThresholdAdder(threshold: Double) extends Serializable {
  def binarize(probPred: Double, prevLabel: Int): Int = if (prevLabel == 0) { //i.e. unlabeled
    if (probPred < threshold) {
      TraditionalPULearner.relNegLabel
    } else {
      TraditionalPULearner.undefLabel
    }
  } else {
    prevLabel // keep as it was (1 or -1)
  }

  val binarizeUDF = udf(binarize(_: Double, _: Int))
}

object TraditionalPULearner {
  val relNegLabel = -1
  val undefLabel = 0
}

case class TraditionalPULearnerConfig(relNegThreshold: Double = 0.5,
                                      maxIters: Int = 1,
                                      classifierConfig: ProbabilisticClassifierConfig = LogisticRegressionConfig()
                                     ) extends PositiveUnlabeledLearnerConfig {
  override def build(): PositiveUnlabeledLearner = {
    classifierConfig match {
      case lrc: LogisticRegressionConfig => new TraditionalPULearner(relNegThreshold, maxIters, lrc.build())
      case rfc: RandomForestConfig => new TraditionalPULearner(relNegThreshold, maxIters, rfc.build())
    }
  }
}
