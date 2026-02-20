/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shm_source.hpp"

#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace lfs::vis::shm;

static const char* SHM_SOURCE_ID = "lichtfeld_shm_source";
static const char* DEFAULT_SHM_NAME = "/LichtFeld-Studio";

static void shm_close(LichtfeldShmSource* ctx) {
    if (ctx->header) {
        munmap(ctx->header, ctx->shm_size);
        ctx->header = nullptr;
    }
    if (ctx->shm_fd >= 0) {
        close(ctx->shm_fd);
        ctx->shm_fd = -1;
    }
    ctx->shm_size = 0;
}

static bool shm_open_segment(LichtfeldShmSource* ctx) {
    shm_close(ctx);

    ctx->shm_fd = shm_open(ctx->shm_name, O_RDWR, 0);
    if (ctx->shm_fd < 0)
        return false;

    ShmHeader probe;
    if (read(ctx->shm_fd, &probe, sizeof(probe)) != sizeof(probe)) {
        close(ctx->shm_fd);
        ctx->shm_fd = -1;
        return false;
    }

    if (probe.magic != SHM_MAGIC || probe.version != SHM_VERSION) {
        close(ctx->shm_fd);
        ctx->shm_fd = -1;
        return false;
    }

    ctx->shm_size = shmTotalSize(probe.max_width, probe.max_height);

    void* mapped = mmap(nullptr, ctx->shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->shm_fd, 0);
    if (mapped == MAP_FAILED) {
        close(ctx->shm_fd);
        ctx->shm_fd = -1;
        ctx->shm_size = 0;
        return false;
    }

    ctx->header = static_cast<ShmHeader*>(mapped);
    ctx->last_frame_id = 0;
    return true;
}

static const char* lfs_get_name(void*) {
    return "LichtFeld Studio";
}

static void* lfs_create(obs_data_t* settings, obs_source_t* source) {
    auto* ctx = new LichtfeldShmSource();
    std::memset(ctx, 0, sizeof(*ctx));
    ctx->source = source;
    ctx->shm_fd = -1;

    const char* name = obs_data_get_string(settings, "shm_name");
    if (!name || !*name)
        name = DEFAULT_SHM_NAME;
    std::strncpy(ctx->shm_name, name, sizeof(ctx->shm_name) - 1);

    return ctx;
}

static void lfs_destroy(void* data) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);

    obs_enter_graphics();
    if (ctx->texture) {
        gs_texture_destroy(ctx->texture);
    }
    obs_leave_graphics();

    shm_close(ctx);
    delete ctx;
}

static uint32_t lfs_get_width(void* data) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);
    return ctx->tex_width > 0 ? static_cast<uint32_t>(ctx->tex_width) : 1920;
}

static uint32_t lfs_get_height(void* data) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);
    return ctx->tex_height > 0 ? static_cast<uint32_t>(ctx->tex_height) : 1080;
}

static void lfs_update(void* data, obs_data_t* settings) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);
    const char* name = obs_data_get_string(settings, "shm_name");
    if (!name || !*name)
        name = DEFAULT_SHM_NAME;

    if (std::strcmp(ctx->shm_name, name) != 0) {
        std::strncpy(ctx->shm_name, name, sizeof(ctx->shm_name) - 1);
        shm_close(ctx);
    }
}

static obs_properties_t* lfs_get_properties(void*) {
    obs_properties_t* props = obs_properties_create();
    obs_properties_add_text(props, "shm_name", "Shared Memory Name", OBS_TEXT_DEFAULT);
    return props;
}

static void lfs_get_defaults(obs_data_t* settings) {
    obs_data_set_default_string(settings, "shm_name", DEFAULT_SHM_NAME);
}

static void lfs_video_tick(void* data, float) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);

    if (!ctx->header) {
        shm_open_segment(ctx);
        if (!ctx->header)
            return;
    }

    const uint64_t current_frame = ctx->header->writer_frame_id.load(std::memory_order_acquire);
    if (current_frame == ctx->last_frame_id) {
        if (++ctx->stale_ticks > 120) {
            shm_close(ctx);
            ctx->stale_ticks = 0;
        }
        return;
    }
    ctx->stale_ticks = 0;

    const uint32_t slot_idx = ctx->header->latest_slot.load(std::memory_order_acquire);
    if (slot_idx >= SHM_NUM_SLOTS)
        return;

    ctx->header->reader_active_slot.store(slot_idx, std::memory_order_release);

    const auto& slot = ctx->header->slots[slot_idx];
    if (slot.width <= 0 || slot.height <= 0)
        return;

    const uint8_t* pixel_data = shmSlotData(ctx->header, static_cast<int>(slot_idx),
                                            ctx->header->max_width, ctx->header->max_height);

    obs_enter_graphics();

    if (ctx->texture && (ctx->tex_width != slot.width || ctx->tex_height != slot.height)) {
        gs_texture_destroy(ctx->texture);
        ctx->texture = nullptr;
    }

    if (!ctx->texture) {
        ctx->texture = gs_texture_create(
            static_cast<uint32_t>(slot.width), static_cast<uint32_t>(slot.height),
            GS_RGBA, 1, &pixel_data, GS_DYNAMIC);
        ctx->tex_width = slot.width;
        ctx->tex_height = slot.height;
    } else {
        gs_texture_set_image(ctx->texture, pixel_data,
                             static_cast<uint32_t>(slot.stride), false);
    }

    obs_leave_graphics();

    ctx->header->reader_active_slot.store(SHM_NUM_SLOTS, std::memory_order_release);
    ctx->last_frame_id = current_frame;
}

static void lfs_video_render(void* data, gs_effect_t*) {
    auto* ctx = static_cast<LichtfeldShmSource*>(data);
    if (!ctx->texture)
        return;

    gs_blend_state_push();
    gs_blend_function(GS_BLEND_SRCALPHA, GS_BLEND_INVSRCALPHA);

    gs_effect_t* effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_technique_t* tech = gs_effect_get_technique(effect, "Draw");
    gs_technique_begin(tech);
    gs_technique_begin_pass(tech, 0);
    obs_source_draw(ctx->texture, 0, 0, 0, 0, false);
    gs_technique_end_pass(tech);
    gs_technique_end(tech);

    gs_blend_state_pop();
}

struct obs_source_info lichtfeld_shm_source = {
    .id = SHM_SOURCE_ID,
    .type = OBS_SOURCE_TYPE_INPUT,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
    .get_name = lfs_get_name,
    .create = lfs_create,
    .destroy = lfs_destroy,
    .get_width = lfs_get_width,
    .get_height = lfs_get_height,
    .get_defaults = lfs_get_defaults,
    .get_properties = lfs_get_properties,
    .update = lfs_update,
    .video_tick = lfs_video_tick,
    .video_render = lfs_video_render,
};
