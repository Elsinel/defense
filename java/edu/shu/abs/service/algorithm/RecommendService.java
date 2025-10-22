package edu.shu.abs.service.algorithm;

import com.alibaba.fastjson.JSONObject;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.shu.abs.common.ColdStartWeightGeneratorUtils;
import edu.shu.abs.common.HttpUtils;
import edu.shu.abs.common.TimeUtils;
import edu.shu.abs.common.authentication.UserInfo;
import edu.shu.abs.common.exception.exception.ServiceException;
import edu.shu.abs.entity.Work;
import edu.shu.abs.service.WorkService;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.math.NumberUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
public class RecommendService {
	@Value("${algorithm-backend.url}")
	private String baseUrl;
	
	
	@Autowired
	private WorkService workService;
	@Autowired
	private RedisTemplate<String, Object> redisTemplate;
	
	
	public List<Work> getRecommendWork() {
		Long userid = UserInfo.getUserId();
		String redisTokenKey = "token4auth:" + userid;
		List<Map<String,String>> list = null;
		if (Boolean.FALSE.equals(redisTemplate.hasKey(redisTokenKey))) {
			list = getTilte(userid);
			redisTemplate.opsForHash().putIfAbsent(redisTokenKey, "user_title", list);
			// 设置过期时间，避免缓存永久有效
			redisTemplate.expire(redisTokenKey, 1, TimeUnit.HOURS);
		}else{
			Object rawUserId = redisTemplate.opsForHash().get(redisTokenKey, "user_title");
			if (rawUserId == null) {
				list = getTilte(userid);
				redisTemplate.opsForHash().putIfAbsent(redisTokenKey, "user_title", list);
				// 设置过期时间，避免缓存永久有效
				redisTemplate.expire(redisTokenKey, 1, TimeUnit.HOURS);
			}else{
				// 校验数据类型，避免类型转换异常
				if (rawUserId instanceof List<?>) {
					// 安全转换类型
					list = convertToListOfMaps(rawUserId);
				} else {
					// 类型不匹配时重新加载数据
					list = getTilte(userid);
					redisTemplate.opsForHash().put(redisTokenKey, "user_title", list);
					redisTemplate.expire(redisTokenKey, 1, TimeUnit.HOURS);
				}
			}
			
		}
		ArrayList<Work> arrayList = new ArrayList<>();
		
		for (Map<String,String> map : list) {
			String title = map.get("title");
			String text = map.get("text");
			String image = map.get("image");
			Work work = workService.getOneByTitle(title);
			work.setText_weight( text);
			work.setImg_weight(image);
			arrayList.add(work);
		}
		return arrayList;
	}
	
	
	public String getTrain(){
		String url = baseUrl + "/api/train";
		String post = null;
		try {
			post = HttpUtils.post(url);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return post;
	}
	
	//可以定时进行执行
	public String getIncTrain(String hour){
		String url = baseUrl + "/api/incremental_train";
		JSONObject tt = new JSONObject();
		tt.put("time", TimeUtils.getFullDate());
		String post = null;
		try {
			post = HttpUtils.post(url,tt.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		return post;
	}
	
	
	public List<Map<String,String>> getTilte (Long user_id) {
		JSONObject tt = new JSONObject();
		tt.put("user_id", user_id);
		tt.put("top_n", 10);
		String url = baseUrl + "/api/recommend";
		String post = null;
		try {
			post = HttpUtils.post(url, tt.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			// 配置ObjectMapper，忽略NaN（解决非法JSON问题）
			ObjectMapper objectMapper = new ObjectMapper();
			objectMapper.configure(JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS, true);
			
			// 解析JSON为实体类
			JsonNode rootNode = objectMapper.readTree(post);
			
			
			List<Map<String,String>> mapList = new ArrayList<>();
			// 提取所有title
			if ("success".equals(rootNode.get("status").asText())) {
				// 获取recommendations数组
				JsonNode recommendationsNode = rootNode.get("recommendations");
				
				// 遍历数组，提取每个对象的title
				for (JsonNode bookNode : recommendationsNode) {
					// 注意：如果字段可能为null，需先判断has("title")
					if (bookNode.has("title")) {
						Map<String,String> map = new HashMap<>();
						String title = bookNode.get("title").asText();
						try{
							String text_contribution = bookNode.get("text_contribution").asText();
							String image_contribution = bookNode.get("image_contribution").asText();
							map.put("text",formatWeightToTwoDecimals(text_contribution)+"");
							map.put("image",formatWeightToTwoDecimals(image_contribution)+"");
						}catch (Exception e){
							double[] doubles = ColdStartWeightGeneratorUtils.generateRandomWeightPair();
							map.put("text",doubles[0]+"");
							map.put("image",doubles[1]+"");
						}
						map.put("title",title);
						mapList.add(map);
					}
				}
			}
			return mapList;
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}
	
	@SuppressWarnings("unchecked")
	private List<Map<String, String>> convertToListOfMaps(Object rawData) {
		List<Map<String, String>> result = new ArrayList<>();
		List<?> rawList = (List<?>) rawData;
		
		for (Object item : rawList) {
			if (item instanceof Map<?, ?>) {
				Map<?, ?> rawMap = (Map<?, ?>) item;
				Map<String, String> stringMap = new HashMap<>();
				
				for (Map.Entry<?, ?> entry : rawMap.entrySet()) {
					if (entry.getKey() instanceof String && entry.getValue() instanceof String) {
						stringMap.put((String) entry.getKey(), (String) entry.getValue());
					}
				}
				result.add(stringMap);
			}
		}
		
		return result;
	}
	
	public static double formatWeightToTwoDecimals(String rawWeight) {
		double aDouble = NumberUtils.toDouble(rawWeight);
		// 使用BigDecimal避免浮点数精度丢失，RoundingMode.HALF_UP表示四舍五入
		return new BigDecimal(aDouble)
				.setScale(2, RoundingMode.HALF_UP)
				.doubleValue();
	}
}
