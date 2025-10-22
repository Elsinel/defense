package edu.shu.abs.common;

import java.text.SimpleDateFormat;
import java.util.Date;

public class TimeUtils {
	// 返回格式化的日期
	public static String getFullDate() {
		String formater = "yyyy-MM-dd";
		SimpleDateFormat format = new SimpleDateFormat(formater);
		Date myDate = new Date();
		return format.format(myDate);
	}
	
	public static void main(String[] args) {
		System.out.println(TimeUtils.getFullDate());
	}
}
