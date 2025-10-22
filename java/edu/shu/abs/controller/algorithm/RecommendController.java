package edu.shu.abs.controller.algorithm;

import edu.shu.abs.common.Result;
import edu.shu.abs.entity.Work;
import edu.shu.abs.service.algorithm.RecommendService;
import edu.shu.abs.vo.algorithm.WorkPredictRatingVo;
import edu.shu.abs.vo.algorithm.recommend.LfmPredictVo;
import edu.shu.abs.vo.algorithm.recommend.LfmRecallQueryVo;
import edu.shu.abs.vo.algorithm.recommend.LfmWorkSimilarQueryVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-04-08 11:05:54
 */
@RestController
@RequestMapping("/recommend")
public class RecommendController {
    @Autowired
    private RecommendService recommendService;
    
    //推荐图书
    @GetMapping("/books")
    public Result getSimilarWork() {
        List<Work> works = recommendService.getRecommendWork();
        return Result.success().data("works", works);
    }
    
    //预测图书
    
    
    //全局训练
    
    @PostMapping("/train")
    public Result train() {
        String train = recommendService.getTrain();
        return Result.success().message(train);
    }
    
    //增量训练
    
    
    @Deprecated
    @PostMapping("/inc_train")
    public Result inc_train() {
        //提前24小时进行训练，走定时处理
        String train = recommendService.getIncTrain("24");
        return Result.success().message(train);
    }
    
    
    
}
