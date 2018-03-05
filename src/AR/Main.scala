package AR

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.sql.SparkSession

// ./spark-submit --master yarn --class AR.Main --executor-memory 20G --driver-memory 20G /opt/lg/AR.jar hdfs:///user/lg/AR hdfs:///user/lg/AR
// /usr/local/hadoop/spark-2.1.2-bin-hadoop2.7/bin/spark-submit --master spark://master:7077 --executor-memory 20G --driver-memory 20G --total-executor-cores 80 /home/xjtudlc/AR/AR.jar hdfs:///user/xjtudlc/AR hdfs:///user/xjtudlc/AR/ hdfs:///user/xjtudlc/AR/


/**
  * Created by lg on 2017/12/26.
  */
object Main {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName(this.getClass.getSimpleName).getOrCreate()

    //1.频繁模式挖掘
    val D = spark.sparkContext.textFile(args(0) + "/D.dat").repartition(128).map(_.split(" ").map(_.toInt)).cache()
    //val D = spark.sparkContext.textFile("/user/lg/AR/minD.dat").map(_.split(" ").map(_.toInt)).cache()
    val minSupport = 0.092
    val numPartition = 128
    val model = new FPGrowth().setMinSupport(minSupport).setNumPartitions(numPartition).run(D)
    model.freqItemsets.map(_.items.mkString(" ")).sortBy(x => x).repartition(1).saveAsTextFile(args(1) + "/Frequent_Pattern")
    println("频繁模式挖掘Done!")
    //model.freqItemsets.map(_.items.mkString(" ")).sortBy(_.split(" ")).repartition(1).saveAsTextFile("/user/lg/AR/Frequent_Pattern")
    /*
        println(s"Number of frequent itemsets: ${model.freqItemsets.count()}") //查看频繁模式的数量
        model.freqItemsets.collect().foreach { itemset =>
          println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
        }
    */


    //2.关联规则生成
    val minConfidence = 0.8
    val rule = model.generateAssociationRules(minConfidence).collect()
    val bcRule = spark.sparkContext.broadcast(rule)
    println("关联规则生成Done!")
    /*
        println(model.generateAssociationRules(minConfidence).collect().length) //查看规则生成的数量
        rule.foreach(rule => {
          println(rule.antecedent.mkString(",") + "-->" +
            rule.consequent.mkString(",") + "-->" + rule.confidence)
        })
    */

    //3.关联规则匹配 4.推荐分值计算
    val U = spark.sparkContext.textFile(args(0) + "/U.dat").repartition(128).map(_.split(" ").map(_.toInt)).cache()
    //   val U = spark.sparkContext.textFile("/user/lg/AR/minU.dat").map(_.split(" ").map(_.toInt)).cache()
    val recommend = U.map { Tu =>
      var highest_confidence = 0D
      var recommendItem = 0;
      bcRule.value.foreach { r =>
        if (r.antecedent.toSet.subsetOf(Tu.toSet) && (!Tu.contains(r.consequent.head))) {
          //给出置信度最大的项,如果置信度最大的项有多个，则给出编号最小的项
          if (r.confidence > highest_confidence || r.confidence == highest_confidence && recommendItem != 0 && r.consequent.head < recommendItem) {
            highest_confidence = r.confidence
            recommendItem = r.consequent.head
          }
        }
      }
      recommendItem
    }
    println("关联规则匹配及推荐Done!")
    recommend.repartition(1).saveAsTextFile(args(1) + "/Recommend")
    //   recommend.repartition(1).saveAsTextFile("/user/lg/AR/Recommend")

  }
}
