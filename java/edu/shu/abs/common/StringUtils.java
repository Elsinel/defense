package edu.shu.abs.common;

import edu.shu.abs.common.exception.exception.ServiceException;
import org.springframework.data.redis.core.RedisTemplate;

import javax.servlet.http.HttpSession;

public class StringUtils {
	
	private static final String LOCAL_IMAGE_ROOT = "/images/downloaded_images/";
	
	public static Long getUserId(HttpSession session, RedisTemplate redisTemplate) {
		Object token = session.getAttribute("token");
		String redisTokenKey = "token4auth:" + token;
		// 检查token是否存在
		if (Boolean.FALSE.equals(redisTemplate.hasKey(redisTokenKey))) {
			throw new ServiceException("token不存在或错误");
		}
		
		// 检查字段是否存在
		Long rawUserId = (Long) redisTemplate.opsForHash().get(redisTokenKey, "user_id");
		return rawUserId;
	}
	
	
	public static String getCoverLink(String link){
		if(link==null){
			return "/images/no_pic.png";
		}
		int lastSlashIndex = link.lastIndexOf('/');
		if (lastSlashIndex == -1 || lastSlashIndex == link.length() - 1) {
			return "/images/no_pic.png";
		}
		
		String fileName = link.substring(lastSlashIndex + 1);
		
		// 构建本地路径
		return LOCAL_IMAGE_ROOT + fileName;
	
	}
	
}
