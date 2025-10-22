package edu.shu.abs.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Result;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.entity.ReviewUserWork;
import edu.shu.abs.service.ReviewUserWorkService;
import edu.shu.abs.vo.review.ReviewNewPostVo;
import edu.shu.abs.vo.review.ReviewQueryConditionVo;
import edu.shu.abs.vo.review.ReviewUserWorkWithUserVo;
import edu.shu.abs.vo.review.ReviewUserWorkWithWorkVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpSession;
import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@RestController
@RequestMapping("/review")
public class ReviewUserWorkController {
    @Autowired
    private ReviewUserWorkService reviewUserWorkService;

    @GetMapping("/{workId}")
    public Result getPage(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize,
                          @PathVariable long workId, ReviewQueryConditionVo condition) {
        IPage<ReviewUserWorkWithUserVo> page = reviewUserWorkService.getPage(currentPage, pageSize, workId, condition);
        return Result.success().data("page", page);
    }

    @GetMapping("/my")
    public Result getMyPageReview(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, HttpSession session) {
        IPage<ReviewUserWorkWithWorkVo> reviewPage = reviewUserWorkService.getMyPageReview(currentPage, pageSize);
        reviewPage.getRecords().forEach(review -> review.setCoverLink(StringUtils.getCoverLink(review.getCoverLink())));
        return Result.success().data("page", reviewPage);
    }
    
    @GetMapping("/all")
    public Result getMyPageCollection(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, HttpSession session) {
        IPage<ReviewUserWorkWithWorkVo> reviewPage = reviewUserWorkService.getAllPageReview(currentPage, pageSize);
        reviewPage.getRecords().forEach(review -> review.setCoverLink(StringUtils.getCoverLink(review.getCoverLink())));
        return Result.success().data("page", reviewPage);
    }
    
    
    @GetMapping("/knowledge-graph")
    public Result getKnowledgeGraph() {
        List<ReviewUserWorkWithWorkVo> reviewPageList = reviewUserWorkService.getMyAllCollectionWithCheckingWork();
        reviewPageList.forEach(review -> review.setCoverLink(StringUtils.getCoverLink(review.getCoverLink())));
        return Result.success().data("list", reviewPageList);
    }
    

    @GetMapping("/my/{workId}")
    public Result getMyReview(@PathVariable long workId) {
        ReviewUserWork review = reviewUserWorkService.getMyReview(workId);
        return Result.success().data("one", review);
    }

    @PostMapping
    public Result updateMyReview(@RequestBody ReviewNewPostVo review) {
        boolean res = reviewUserWorkService.updateReview(review);
        return res ? Result.success().message("评论发表成功") : Result.error();
    }

    @GetMapping("/visit/{userId}")
    public Result getOtherPageReview(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, @PathVariable long userId) {
        IPage<ReviewUserWorkWithWorkVo> reviewPage = reviewUserWorkService.getOtherPageReview(currentPage, pageSize, userId);
        return Result.success().data("page", reviewPage);
    }
}
