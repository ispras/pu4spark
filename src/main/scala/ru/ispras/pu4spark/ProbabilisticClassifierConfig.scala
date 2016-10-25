/*
 * Copyright 2016 ISP RAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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