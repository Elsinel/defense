package edu.shu.abs.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.Page;
import edu.shu.abs.common.StringUtils;
import edu.shu.abs.common.authentication.UserInfo;
import edu.shu.abs.common.base.BaseServiceImpl;
import edu.shu.abs.common.exception.exception.NoAccessException;
import edu.shu.abs.common.exception.exception.NotExistException;
import edu.shu.abs.common.exception.exception.ServiceException;
import edu.shu.abs.entity.RecordCollectionWork;
import edu.shu.abs.entity.Work;
import edu.shu.abs.mapper.RecordCollectionWorkMapper;
import edu.shu.abs.mapper.WorkMapper;
import edu.shu.abs.service.RecordTagWorkService;
import edu.shu.abs.service.TagService;
import edu.shu.abs.service.WorkService;
import edu.shu.abs.vo.work.*;
import lombok.SneakyThrows;
import org.apache.commons.lang3.ObjectUtils;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.net.InetAddress;
import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
@Service
public class WorkServiceImpl extends BaseServiceImpl<WorkMapper, Work> implements WorkService {
    @Value("${server.port}")
    private int port;

    @Autowired
    private WorkMapper workMapper;

    @Autowired
    private RecordCollectionWorkMapper recordCollectionWorkMapper;

    @Autowired
    private RecordTagWorkService recordTagWorkService;

    @Autowired
    private TagService tagService;

    @Override
    public IPage<Work> getPage(int currentPage, int pageSize, WorkQueryConditionVo condition) {
        QueryWrapper<Work> wrapper = new QueryWrapper<>();
        /*wrapper.like(ObjectUtils.isNotEmpty(condition.getWorkName()), "work_name", condition.getWorkName())
                .like(ObjectUtils.isNotEmpty(condition.getAuthor()), "author", condition.getAuthor())
                .like(ObjectUtils.isNotEmpty(condition.getPublisher()), "publisher", condition.getPublisher())
                .like(ObjectUtils.isNotEmpty(condition.getIntroduction()), "introduction", condition.getIntroduction())
                .eq("is_deleted", false);*/
        wrapper.eq("is_deleted", false)
                .and(wq -> { // 使用 AND 包裹 OR 条件组
                    boolean isFirst = true;
                    // 处理 work_name
                    if (ObjectUtils.isNotEmpty(condition.getWorkName())) {
                        wq.like("work_name", condition.getWorkName());
                        isFirst = false;
                    }
                    // 处理 author
                    if (ObjectUtils.isNotEmpty(condition.getAuthor())) {
                        if (isFirst) {
                            wq.like("author", condition.getAuthor());
                            isFirst = false;
                        } else {
                            wq.or().like("author", condition.getAuthor());
                        }
                    }
                    // 处理 publisher
                    if (ObjectUtils.isNotEmpty(condition.getPublisher())) {
                        if (isFirst) {
                            wq.like("publisher", condition.getPublisher());
                            isFirst = false;
                        } else {
                            wq.or().like("publisher", condition.getPublisher());
                        }
                    }
                    // 处理 introduction
                    if (ObjectUtils.isNotEmpty(condition.getIntroduction())) {
                        if (isFirst) {
                            wq.like("introduction", condition.getIntroduction());
                        } else {
                            wq.or().like("introduction", condition.getIntroduction());
                        }
                    }
                });
        if (ObjectUtils.isNotEmpty(condition.getTags())) {  // 对tags按空格切分进行多标签查询
            String[] tags = condition.getTags().trim().split("\\s+");
            for (String tag : tags) {
                wrapper.like("tags", tag);
            }
        }
        Page<Work> page = new Page<>(currentPage, pageSize);
        return workMapper.selectPage(page, wrapper);
    }

    @SneakyThrows
    @Override
    public Work getOneDetail(long workId) {
        Work work = workMapper.selectById(workId);
        if (!existWork(workId))
            throw new NotExistException("不存在id=" + workId + "的作品");
        return work;
    }
    
    
    
    @SneakyThrows
    @Override
    public Work getOneByTitle(String title) {
        Work work = workMapper.selectByTitle(title);
        work.setCoverLink(StringUtils.getCoverLink(work.getCoverLink()));
        return work;
    }
    

    @Override
    @Transactional
    public Boolean saveWork(WorkNewPostVo workNewPostVo) {
        if (!UserInfo.isAdmin())
            throw new NoAccessException("只有管理员才能使用此接口");

        if (ObjectUtils.isEmpty(workNewPostVo.getWorkName()))
            throw new ServiceException("作品名不能为空");
        Work work = new Work();
        BeanUtils.copyProperties(workNewPostVo, work);
        boolean res = workMapper.insert(work) > 0;
        return res;
    }

    @Override
    @Transactional
    public Boolean updateWork(WorkEditVo workEditVo) {
        if (!UserInfo.isAdmin())
            throw new NoAccessException("只有管理员才能使用此接口");

        if (workEditVo.getWorkId() == null)
            throw new ServiceException("作品id不能为空");
        proveExistWork(workEditVo.getWorkId());
        if (ObjectUtils.isEmpty(workEditVo.getWorkName()))
            throw new ServiceException("作品名不能为空");

        Work work = getByIdNotNull(workEditVo.getWorkId());
        BeanUtils.copyProperties(workEditVo, work);
        if (work.getTags() != null)
            work.setTags(work.getTags().trim().replaceAll("\\s+", " "));
        boolean res = workMapper.updateById(work) > 0;
        return res;
    }

    /**
     * 删除作品只是逻辑删除<br/>
     * 不需要清除评价关系、访问关系与作品标签关系<br/>
     * 但需要清除收藏关系
     */
    @Override
    @Transactional
    public Boolean dropWork(long workId) {
        if (!UserInfo.isAdmin())
            throw new NoAccessException("只有管理员才能使用此接口");

        Work work = getByIdNotNull(workId);
        if (work.getIsDeleted()) {
            throw new ServiceException("不能重复对同一作品进行逻辑删除");
        }

        work.setIsDeleted(true);
        return workMapper.updateById(work) > 0;
    }

    @Override
    public IPage<WorkRatingVo> getHighestRating(int currentPage, int pageSize) {
        return workMapper.selectHighestRating(new Page<>(currentPage, pageSize));
    }

    @Override
    public IPage<WorkRatingVo> getMostRating(int currentPage, int pageSize) {
        return workMapper.selectMostRatingPage(new Page<>(currentPage, pageSize));
    }

    @Override
    public List<WorkRatingVo> getMostRating(int num) {
        return workMapper.selectMostRatingLimitNum(num);
    }

    @Override
    public IPage<WorkVisitCountVo> getMostVisit(int currentPage, int pageSize) {
        return workMapper.selectMostVisit(new Page<>(currentPage, pageSize));
    }

    @Override
    public IPage<WorkCollectCountVo> getMostCollect(int currentPage, int pageSize) {
        return workMapper.selectMostCollect(new Page<>(currentPage, pageSize));
    }

    /**
     * 返回作品是否存在
     */
    @Override
    public Boolean existWork(long workId) {
        Work work = workMapper.selectById(workId);
        return work != null && !work.getIsDeleted();
    }

    /**
     * 保证文学作品存在
     */
    @Override
    public void proveExistWork(long workId) {
        if (!existWork(workId))
            throw new NotExistException(workId, "文学作品");
    }
}
