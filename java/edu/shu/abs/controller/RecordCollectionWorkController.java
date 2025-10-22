package edu.shu.abs.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Page;
import edu.shu.abs.common.Result;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.entity.Collection;
import edu.shu.abs.entity.Work;
import edu.shu.abs.service.CollectionService;
import edu.shu.abs.service.RecordCollectionWorkService;
import edu.shu.abs.vo.collection.CollectionNewPostVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * <p>
 *  前端控制器
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@RestController
@RequestMapping("/record_collection")
public class RecordCollectionWorkController {
    @Autowired
    private RecordCollectionWorkService recordCollectionWorkService;
    @Autowired
    private CollectionService collectionService;

    @GetMapping("/work")
    public Result getPageCollectionRecord(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize) {
        //如果没有，则创建，算是我的书架的功能
        List<Collection> allCollection = collectionService.getMyAllCollection();
        if(allCollection.isEmpty()){
            //创建一个
            CollectionNewPostVo collectionNewPostVo = new CollectionNewPostVo();
            collectionNewPostVo.setCollectionName(new Date().toString());
            collectionNewPostVo.setIntroduction("本地书架");
            collectionNewPostVo.setIsPublic(true);
            collectionNewPostVo.setIsDefaultCollection(true);
            collectionService.saveNewCollection(collectionNewPostVo);
        }
        List<Collection> allCollectionList = collectionService.getMyAllCollection();
        Long id = allCollectionList.get(0).getCollectionId();
        Page<Work> page = recordCollectionWorkService.getMyRecordPage(currentPage, pageSize, id);
        page.getRecords().forEach(work -> {
            work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        });
        return Result.success().data("page", page);
    }

    @GetMapping("/with_work/{workId}")
    public Result getMyAllCollectionWithCheckingWork(@PathVariable Long workId) {
        List<Map<String, Object>> all = recordCollectionWorkService.getMyAllCollectionWithCheckingWork(workId);
        return Result.success().data("all", all);
    }

    @PostMapping("/favorite/{workId}")
    public Result postNewRecord(@PathVariable Long workId) {
        List<Collection> allCollection = collectionService.getMyAllCollection();
        if(allCollection.isEmpty()){
            //创建一个
            CollectionNewPostVo collectionNewPostVo = new CollectionNewPostVo();
            collectionNewPostVo.setCollectionName(new Date().toString());
            collectionNewPostVo.setIntroduction("本地书架");
            collectionNewPostVo.setIsPublic(true);
            collectionNewPostVo.setIsDefaultCollection(true);
            collectionService.saveNewCollection(collectionNewPostVo);
        }
        List<Collection> allCollectionList = collectionService.getMyAllCollection();
        Long id = allCollectionList.get(0).getCollectionId();
        boolean res = recordCollectionWorkService.saveNewRecord(id, workId);
        return res ? Result.success().message("收藏成功") : Result.error();
    }

    @PostMapping("/default/{workId}")
    public Result postNewRecordIntoDefault(@PathVariable Long workId) {
        boolean res = recordCollectionWorkService.saveNewRecordIntoDefault(workId);
        return res ? Result.success().message("收藏成功") : Result.error();
    }
    
    @DeleteMapping("/delete/{workId}")
    public Result dropRecord(@PathVariable Long workId) {
        List<Collection> allCollectionList = collectionService.getMyAllCollection();
        Long id = allCollectionList.get(0).getCollectionId();
        boolean res = recordCollectionWorkService.dropRecord(id, workId);
        return res ? Result.success().message("移除收藏成功") : Result.error();
    }

 
    @GetMapping("/visit/{collectionId}")
    public Result getOtherPageReview(@RequestParam(defaultValue = "1") int currentPage, @RequestParam(defaultValue = "10") int pageSize, @PathVariable long collectionId) {
        IPage<Work> page = recordCollectionWorkService.getOtherRecordPage(currentPage, pageSize, collectionId);
        return Result.success().data("page", page);
    }
}
