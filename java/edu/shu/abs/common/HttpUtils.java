package edu.shu.abs.common;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.shu.abs.entity.Book;
import edu.shu.abs.entity.RecommendResponse;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class HttpUtils {
	private static final CloseableHttpClient httpClient = HttpClients.createDefault();
	
	/**
	 * 发送POST请求（无请求体）
	 * @param url 请求URL
	 * @return 响应字符串
	 * @throws IOException 可能的IO异常
	 */
	public static String post(String url) throws IOException {
		return post(url, null);
	}
	
	/**
	 * 发送POST请求（带JSON请求体）
	 * @param url 请求URL
	 * @param jsonBody JSON格式的请求体
	 * @return 响应字符串
	 * @throws IOException 可能的IO异常
	 */
	public static String post(String url, String jsonBody) throws IOException {
		HttpPost httpPost = new HttpPost(url);
		httpPost.setHeader("Content-Type", "application/json; charset=UTF-8");
		
		// 设置请求体
		if (jsonBody != null && !jsonBody.isEmpty()) {
			StringEntity entity = new StringEntity(jsonBody, StandardCharsets.UTF_8);
			httpPost.setEntity(entity);
		}
		
		// 执行请求并处理响应
		try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
			HttpEntity responseEntity = response.getEntity();
			if (responseEntity != null) {
				return EntityUtils.toString(responseEntity, StandardCharsets.UTF_8);
			}
			return null;
		}
	}
	
	/**
	 * 关闭HTTP客户端
	 * @throws IOException 可能的IO异常
	 */
	public static void close() throws IOException {
		httpClient.close();
	}
	
	public static void main(String[] args) throws IOException {
		
		JSONObject tt = new JSONObject();
		tt.put("user_id", 1110799);
		tt.put("top_n", 10);
		String url = "http://localhost:5000/api/recommend";
		
		// 构建请求体
		Map<String, Object> requestBody = new HashMap<>();
		requestBody.put("user_id", 1110799);
		requestBody.put("top_n", 10);
		
		// 转换为JSON字符串
		String jsonBody = tt.toString();
		String post = HttpUtils.post(url, jsonBody);
		
		System.out.println(post);
		System.out.println("-----------------------------");
		
		try {
			// 配置ObjectMapper，忽略NaN（解决非法JSON问题）
			ObjectMapper objectMapper = new ObjectMapper();
			objectMapper.configure(JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS, true);
			
			// 解析JSON为实体类
			RecommendResponse response = objectMapper.readValue(post, RecommendResponse.class);
			
			// 提取所有title
			if ("success".equals(response.getStatus()) && response.getRecommendations() != null) {
				List<String> titles = response.getRecommendations().stream()
						.map(Book::getTitle)
						.collect(Collectors.toList());
				
				// 输出结果
				System.out.println("提取的title列表：");
				titles.forEach(System.out::println);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
 
	
}

