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

import org.apache.spark.sql.DataFrame

/**
  * Performs positive unlabeled (PU) learning, i.e. training a binary classifier in a semi-supervised way
  * from only positive and unlabeled examples
  *
  * @author Nikita Astrakhantsev (astrakhantsev@ispras.ru)
  */
trait PositiveUnlabeledLearner {

  /**
    * Updates dataframe by applying positive-unlabeled learning (append column with result of classification).
    *
    * @param df dataframe containing, among others, column with labels and features to be used in PU-learning
    * @param labelColumnName name for column containing 1 - positives and 0 - unlabeled marks for each instance
    * @param featuresColumnName name for 1 column containing features array (e.g. after VectorAssembler)
    * @param finalLabel name for column containing labels of final classification (1 for positive and -1 for negatives)
    * @return dataframe with new column corresponding to final classification
    */
  def weight(df: DataFrame,
             labelColumnName: String = "featuresCol",
             featuresColumnName: String = "labelCol",
             finalLabel: String = "finalLabel"): DataFrame
}

/**
  * Subclasses should be case classes in order to be easily serializable (e.g. to JSON)
  */
trait PositiveUnlabeledLearnerConfig {
  def build(): PositiveUnlabeledLearner
}

/**
  * Needed for serialization by json4s (should be passed to org.json4s.ShortTypeHints)
  */
object PositiveUnlabeledLearnerConfig {
  val subclasses = List(classOf[TraditionalPULearnerConfig], classOf[GradualReductionPULearnerConfig])
}