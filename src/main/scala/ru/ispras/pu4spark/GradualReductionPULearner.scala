package ru.ispras.pu4spark

import org.apache.logging.log4j.LogManager
import org.apache.spark.ml.classification.{ProbabilisticClassificationModel, ProbabilisticClassifier}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Modified Positive-Unlabeled learning algorithm; main idea is to gradually refine set of positive examples<br>
  *
  * Pseudocode was taken from:
  *  Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015).
  *  Detecting positive and negative deceptive opinions using PU-learning.
  *  Information Processing & Management, 51(4), 433-443.
  *
  * @author Nikita Astrakhantsev (astrakhantsev@ispras.ru)
  */
class GradualReductionPULearner[
    E <: ProbabilisticClassifier[Vector, E, M],
    M <: ProbabilisticClassificationModel[Vector, M]](
        relNegThreshold: Double,
        classifier: ProbabilisticClassifier[Vector, E, M]) extends TwoStepPULearner[E,M](classifier) {

  val log = LogManager.getLogger(getClass)

  override def weight(df: DataFrame, labelColumnName: String, featuresColumnName: String, finalLabel: String): DataFrame = {
    val oneStepPUDF: DataFrame = zeroStep(df, labelColumnName, featuresColumnName, finalLabel)
      .drop("probability").drop("prediction").drop("rawPrediction").drop(ProbabilisticClassifierConfig.labelName)

    val prevLabel = "prevLabel"
    val curLabel = "curLabel"
    var curDF = replaceZerosByUndefLabel(oneStepPUDF, labelColumnName, prevLabel, GradualReductionPULearner.undefLabel)

    val confAdder = new GradRelNegConfidenceThresholdAdder(relNegThreshold, GradualReductionPULearner.undefLabel)

    //replace weights by binary column for further learning (induce labels for curLabDF)
    val curLabelColumn = confAdder.binarizeUDF(curDF(finalLabel), curDF(prevLabel))

    curDF = curDF.withColumn(curLabel, curLabelColumn).cache()
    var newRelNegCount = curDF
      //unlabeled in previous iterations && negative in current iteration
      .filter(curDF(prevLabel) === GradualReductionPULearner.undefLabel && curDF(curLabel) === GradualReductionPULearner.relNegLabel)
      .count()

    log.debug(s"newRelNegCount: $newRelNegCount")
    var prevNewRelNegCount = newRelNegCount
    val totalPosCount = curDF.filter(curDF(curLabel) === GradualReductionPULearner.posLabel).count()
    var totalRelNegCount = curDF.filter(curDF(curLabel) === GradualReductionPULearner.relNegLabel).count()

    var prevGain = Long.MaxValue
    var curGain = newRelNegCount

    do {
      //learn new classifier
      val curLabDF = curDF.filter(curDF(curLabel) !== GradualReductionPULearner.undefLabel)

      val newPreparedDf = indexLabelColumn(curLabDF, curLabel, ProbabilisticClassifierConfig.labelName,
        Seq(GradualReductionPULearner.relNegLabel.toString, GradualReductionPULearner.posLabel.toString))

      val model = classifier.fit(newPreparedDf)

      //apply classifier to all data (however, we are interested in ReliableNegatives data only, see confAdder)
      val labUnlabDF = model.transform(curDF)
      curDF = labUnlabDF.withColumn(finalLabel, getPOne(labUnlabDF("probability")))
        .drop("probability").drop("prediction").drop("rawPrediction").drop(ProbabilisticClassifierConfig.labelName)
      curDF = curDF.drop(prevLabel)
        .withColumnRenamed(curLabel, prevLabel)

      val innerConfAdder = new GradRelNegConfidenceThresholdAdder(relNegThreshold, GradualReductionPULearner.relNegLabel)
      val curLabelColumn = innerConfAdder.binarizeUDF(curDF(finalLabel), curDF(prevLabel))

      curDF = curDF.withColumn(curLabel, curLabelColumn).cache()
      prevNewRelNegCount = newRelNegCount
      newRelNegCount = curDF
        //negative in current iteration
        .filter(curDF(curLabel) === GradualReductionPULearner.relNegLabel)
        .count()
      totalRelNegCount = curDF.filter(curDF(curLabel) === GradualReductionPULearner.relNegLabel).count()
      prevGain = curGain
      curGain = prevNewRelNegCount - totalRelNegCount
      log.debug(s"newRelNegCount: $newRelNegCount, prevNewRelNegCount: $prevNewRelNegCount, totalRelNegCount: $totalRelNegCount")
      log.debug(s"curGain: $curGain, prevGain: $prevGain")
    } while (curGain > 0 && curGain < prevGain && totalPosCount < totalRelNegCount)
    curDF
  }
}

private class GradRelNegConfidenceThresholdAdder(threshold: Double, labelToConsider: Int) extends Serializable {
  def binarize(probPred: Double, prevLabel: Int): Int = if (prevLabel == labelToConsider) {
    if (probPred < threshold) {
      GradualReductionPULearner.relNegLabel
    } else {
      GradualReductionPULearner.undefLabel
    }
  } else {
    prevLabel // keep as it was //(1 or -1 in case of unlabeled classification)
  }

  val binarizeUDF = udf(binarize(_: Double, _: Int))
}

object GradualReductionPULearner {
  val relNegLabel = 0
  val posLabel = 1
  val undefLabel = -1
}

case class GradualReductionPULearnerConfig(relNegThreshold: Double = 0.5,
                                           classifierConfig: ProbabilisticClassifierConfig) extends PositiveUnlabeledLearnerConfig {
  override def build(): PositiveUnlabeledLearner = {
    classifierConfig match {
      case lrc: LogisticRegressionConfig => new GradualReductionPULearner(relNegThreshold, lrc.build())
      case rfc: RandomForestConfig => new GradualReductionPULearner(relNegThreshold, rfc.build())
    }
  }
}
