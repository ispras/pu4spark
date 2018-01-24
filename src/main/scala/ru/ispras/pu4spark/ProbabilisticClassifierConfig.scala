package ru.ispras.pu4spark

import org.apache.spark.ml.classification._
import org.apache.spark.mllib.linalg.Vector

/**
  * @author Nikita Astrakhantsev (astrakhantsev@ispras.ru)
  */
sealed trait ProbabilisticClassifierConfig

case class LogisticRegressionConfig(maxIter: Int = 100,
                                    regParam: Double = 1.0e-8,
                                    elasticNetParam: Double = 0.0)
  extends ProbabilisticClassifierConfig {
  def build(): ProbabilisticClassifier[Vector, LogisticRegression, LogisticRegressionModel] = {
    new LogisticRegression()
      .setLabelCol(ProbabilisticClassifierConfig.labelName).setFeaturesCol(ProbabilisticClassifierConfig.featuresName)
      .setMaxIter(maxIter).setRegParam(regParam).setElasticNetParam(elasticNetParam)
  }
}

case class RandomForestConfig(numTrees: Int = 512)
  extends ProbabilisticClassifierConfig {
  def build(): ProbabilisticClassifier[Vector, RandomForestClassifier, RandomForestClassificationModel] = {
    new RandomForestClassifier()
      .setLabelCol(ProbabilisticClassifierConfig.labelName).setFeaturesCol(ProbabilisticClassifierConfig.featuresName)
      .setNumTrees(numTrees)
  }
}

object ProbabilisticClassifierConfig {
  val labelName = "label"
  val featuresName = "indexedFeatures"
  val subclasses = List(classOf[LogisticRegressionConfig], classOf[RandomForestConfig])
}
