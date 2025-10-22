package edu.shu.abs.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Result;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.common.authentication.UserInfo;
import edu.shu.abs.service.HistoryUserWorkService;
import edu.shu.abs.vo.history.HistoryVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpSession;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@RestController
@RequestMapping("/history")
public class HistoryUserWorkController {
    @Autowired
    private HistoryUserWorkService historyUserWorkService;
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    @GetMapping("/page")
    public Result getPage(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, HttpSession session) {
        Long userId = UserInfo.getUserId();
        if(userId == null){
            userId = 738173L;
        }
        IPage<HistoryVo> page = historyUserWorkService.getPage(currentPage, pageSize,userId);
        if(page == null){
            return Result.error();
        }
        page.getRecords().forEach(history -> {
            history.setCoverLink(StringUtils.getCoverLink(history.getCoverLink()));
        });
        return Result.success().data("page", page);
    }

    @PostMapping("/{workId}")
    public Result addHistory(@PathVariable long workId) {
        boolean res = historyUserWorkService.updateHistory(workId);
        return res ? Result.success().message("作品访问记录新增成功") : Result.error();
    }
}
