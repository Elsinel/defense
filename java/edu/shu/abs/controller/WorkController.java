package edu.shu.abs.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Result;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.entity.Work;
import edu.shu.abs.service.WorkService;
import edu.shu.abs.vo.work.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-07-24 11:05:54
 */
@RestController
@RequestMapping("/work")
public class WorkController {
    @Autowired
    private WorkService workService;
 
    @GetMapping("/page")
    public Result getPage(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, WorkQueryConditionVo condition) {
        IPage<Work> page = workService.getPage(currentPage, pageSize, condition);
        page.getRecords().forEach(
                work -> work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()))
        );
        return Result.success().data("page", page);
    }
 
    @GetMapping("/{workId}")
    public Result getById(@PathVariable long workId) {
        Work work = workService.getOneDetail(workId);
        work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        return Result.success().data("book", work);
    }
 
    @GetMapping("/exist/{workId}")
    public Result existWork(@PathVariable long workId) {
        Boolean res = workService.existWork(workId);
        return Result.success().data("exist", res);
    }
    
    @GetMapping("/knowledge-graph/{workId}")
    public Result knowledgeGraph(@PathVariable long workId) {
        Boolean res = workService.existWork(workId);
        return Result.success().data("exist", res);
    }
 
    
    
    @PostMapping("/add")
    public Result postWork(@RequestBody WorkNewPostVo workNewPostVo) {
        Boolean res = workService.saveWork(workNewPostVo);
        return res ? Result.success().message("文学作品新增成功") : Result.error();
    }
 
    @PutMapping("/edit")
    public Result editWork(@RequestBody WorkEditVo workEditVo) {
        Boolean res = workService.updateWork(workEditVo);
        return res ? Result.success().message("文学作品修改成功") : Result.error();
    }
 
    @DeleteMapping("/delete/{workId}")
    public Result deleteWork(@PathVariable long workId) {
        Boolean res = workService.dropWork(workId);
        return res ? Result.success().message("文学作品删除成功") : Result.error();
    }
 
    @GetMapping("/highest_rating")
    public Result getHighestRating(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "50") int pageSize) {
        IPage<WorkRatingVo> page = workService.getHighestRating(currentPage, pageSize);
        page.getRecords().forEach(work -> {
            work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        });
        return Result.success().data("page", page);
    }
 
    @GetMapping("/most_rating")
    public Result getMostRating(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "50") int pageSize) {
        IPage<WorkRatingVo> page = workService.getMostRating(currentPage, pageSize);
        page.getRecords().forEach(work -> {
            work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        });
        return Result.success().data("page", page);
    }

    @GetMapping("/most_visit")
    public Result getMostVisit(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "50") int pageSize) {
        IPage<WorkVisitCountVo> page = workService.getMostVisit(currentPage, pageSize);
        page.getRecords().forEach(work -> {
            work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        });
        return Result.success().data("page", page);
    }

    @GetMapping("/most_collect")
    public Result getMostCollect(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "50") int pageSize) {
        IPage<WorkCollectCountVo> page = workService.getMostCollect(currentPage, pageSize);
        return Result.success().data("page", page);
    }
    
    @GetMapping("/recomment")
    public Result getRecommendWork(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize) {
        IPage<WorkVisitCountVo> page = workService.getMostVisit(currentPage, pageSize);
        page.getRecords().forEach(work -> {
            work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        });
        return Result.success().data("page", page);
    }
    
}
