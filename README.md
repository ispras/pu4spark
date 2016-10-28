# pu4spark
A library for [Positive-Unlabeled Learning](https://en.wikipedia.org/wiki/One-class_classification#PU_learning)
for Apache Spark MLlib (ml package)

## Implemented algorithms

### Traditional PU
Original Positive-Unlabeled learning algorithm; firstly proposed in
> Liu, B., Dai, Y., Li, X. L., Lee, W. S., & Philip, Y. (2002).
Partially supervised classification of text documents.
In ICML 2002, Proceedings of the nineteenth international conference on machine learning. (pp. 387–394).

### Gradual Reduction PU (aka PU-LEA)
Modified Positive-Unlabeled learning algorithm;
main idea is to gradually refine set of positive examples.
Pseudocode was taken from:
>Fusilier, D. H., Montes-y-Gómez, M., Rosso, P., & Cabrera, R. G. (2015).
Detecting positive and negative deceptive opinions using PU-learning.
Information Processing & Management, 51(4), 433-443.

## Requirements

Spark 1.5+

(Spark 2+ was not tested,
but should work if replace `SparkContext` by `SparkSession`
and `mllib.linalg.Vector` by `ml.linalg.Vector`)

## Linking

The library is published into Maven central and JCenter.
Add the following lines depending on your build system.

### Gradle

```gradle
compile 'ru.ispras:pu4spark:0.1'
```

### Maven

```xml
<dependency>
    <groupId>ru.ispras</groupId>
    <artifactId>pu4spark</artifactId>
    <version>0.1</version>
</dependency>
```

### SBT

```
libraryDependencies += "ru.ispras" % "pu4spark" % "0.1"
```

## Building from Sources

Build library with gradle:

```shell
./gradlew jar
```

## Usage example


```scala
val inputLabelName = "category"
val srcFeaturesName = "srcFeatures"
val outputLabel = "outputLabel"

val puLearnerConfig = TraditionalPULearnerConfig(0.05, 1, LogisticRegressionConfig())
val puLearner = puLearnerConfig.build()
val df = ... //needed df that contains at least the following columns:
// binary label for positive and unlabel (inputLabelName)
// and features assembled as vector (featuresName)

val weightedDF = puLearner.weight(preparedDf, inputLabelName, srcFeaturesName, outputLabel)
```
Returned dataframe contains probability estimation for each instance in the column `outputLabel`.

Features can be assembled to one column by using [VectorAssembler](https://spark.apache.org/docs/1.6.2/ml-features.html#vectorassembler):
```scala
val assembler = new VectorAssembler()
  .setInputCols(df.columns.filter(c => c != rowName)) //keep here only feature columns
  .setOutputCol(featuresName)
val pipeline = new Pipeline().setStages(Array(assembler))
val preparedDf = pipeline.fit(df).transform(dfForML)
```
