package edu.shu.abs.common;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Random;

public class ColdStartWeightGeneratorUtils {
	// 随机数生成器（默认无种子，每次运行随机；可手动设置种子保证可复现）
	private static Random random = new Random();
	
	/**
	 * 可选：设置随机种子（用于测试，确保每次生成相同的随机权重）
	 * @param seed 随机种子（如12345）
	 */
	public static void setRandomSeed(long seed) {
		random = new Random(seed);
	}
	
	/**
	 * 生成一组冷启动随机权重对（文本权重+图像权重）
	 * @return 权重数组：index[0] = 文本权重，index[1] = 图像权重（均为两位小数，和为1）
	 */
	public static double[] generateRandomWeightPair() {
		// 1. 生成0~100的整数（对应0.00~1.00的两位小数，避免浮点数精度问题）
		int textWeightInt = random.nextInt(101); // 范围：0 ≤ textWeightInt ≤ 100
		// 2. 转换为两位小数的文本权重（如58 → 0.58）
		double textContribution = new BigDecimal(textWeightInt)
				.divide(new BigDecimal(100), 2, RoundingMode.HALF_UP)
				.doubleValue();
		// 3. 计算图像权重（1 - 文本权重，自动保证两位小数）
		double imageContribution = new BigDecimal(1.00)
				.subtract(new BigDecimal(textContribution))
				.setScale(2, RoundingMode.HALF_UP)
				.doubleValue();
		// 4. 返回权重对
		return new double[]{textContribution, imageContribution};
	}
}
