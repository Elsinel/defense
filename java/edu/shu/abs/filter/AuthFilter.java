package edu.shu.abs.filter;

import edu.shu.abs.common.authentication.UserInfo;
import edu.shu.abs.common.exception.exception.ServiceException;
import edu.shu.abs.constant.Role;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import javax.servlet.*;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;

@Component
public class AuthFilter implements Filter {
	
	@Autowired
	private RedisTemplate<String, Object> redisTemplate;
	
	// 需要排除的路径
	private static final String[] EXCLUDE_PATHS = {
			"/user/index",
			"/user/login",
			"/user/toreg",
			"/user/register"
	};
	// 静态资源路径
	private static final String[] STATIC_RESOURCE_PATHS = {
			"/static/", "/css/", "/js/", "/images/", "/fonts/",
			"/img/", "/webjars/", "/favicon.ico"
	};
	
	
	@Override
	public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
			throws IOException, ServletException {
		
		HttpServletRequest httpRequest = (HttpServletRequest) request;
		HttpServletResponse httpResponse = (HttpServletResponse) response;
		String requestURI = httpRequest.getRequestURI();
		
		// 检查是否是静态资源
		if (isStaticResource(requestURI)) {
			chain.doFilter(request, response);
			return;
		}
		
		// 检查是否是需要排除的路径
		if (isExcludedPath(requestURI)) {
			chain.doFilter(request, response);
			return;
		}
		
		
		// 非排除路径，检查session中的token
		HttpSession session = httpRequest.getSession(false);
		if (session == null || session.getAttribute("token") == null) {
			// 没有token，重定向到首页
			httpResponse.sendRedirect("/user/index");
			return;
		}
		
		String token = session.getAttribute("token").toString();
		
		String redisTokenKey = "token4auth:" + token;
		
		if (Boolean.FALSE.equals(redisTemplate.hasKey(redisTokenKey))) {
			throw new ServiceException("token不存在或错误");
		}
		
		// 检查字段是否存在
		Object rawUserId = redisTemplate.opsForHash().get(redisTokenKey, "user_id");
		Object rawRole = redisTemplate.opsForHash().get(redisTokenKey, "role");
		if (rawUserId == null || rawRole == null) {
			throw new ServiceException("token属性不完整");
		}
		
		// 写入UserInfo (ThreadLocal)
		Long userId = Long.parseLong(rawUserId.toString());
		Role role = Role.getRole(Integer.parseInt(rawRole.toString()));
		UserInfo.set(userId, role, token);   // 写入userId到线程副本
		
		// 有token，继续处理
		chain.doFilter(request, response);
	}
	
	// 判断是否为静态资源
	private boolean isStaticResource(String requestURI) {
		for (String path : STATIC_RESOURCE_PATHS) {
			if (requestURI.startsWith(path)) {
				return true;
			}
		}
		return false;
	}
	
	// 判断是否为排除路径
	private boolean isExcludedPath(String requestURI) {
		for (String path : EXCLUDE_PATHS) {
			if (requestURI.startsWith(path)) {
				return true;
			}
		}
		return false;
	}
	
	@Override
	public void init(FilterConfig filterConfig) throws ServletException {
		// 初始化操作，如需要可添加
	}
	
	@Override
	public void destroy() {
		// 销毁操作，如需要可添加
	}
}
