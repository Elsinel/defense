package edu.shu.abs.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import edu.shu.abs.common.base.BaseService;
import edu.shu.abs.entity.HistoryUserWork;
import edu.shu.abs.vo.history.HistoryVo;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author zyh
 * @since 2025-05-26 11:05:54
 */
public interface HistoryUserWorkService extends BaseService<HistoryUserWork> {
    IPage<HistoryVo> getPage(int currentPage, int pageSize);
    
    IPage<HistoryVo> getPage(int currentPage, int pageSize, long userId);

    boolean updateHistory(long workId);

}
