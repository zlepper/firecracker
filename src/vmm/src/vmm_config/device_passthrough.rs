// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::pci::PciSBDF;

use std::path::{Path, PathBuf};

/// Errors for device passthrough configuration.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DevicePassthroughConfigError {
    /// Duplicate device passthrough source: {0}
    DuplicateSource(String),
    /// Invalid device passthrough SBDF: {0}
    InvalidSBDF(String),
    /// A passthrough device needs exactly one of `sbdf` or both `group_path` and `device`
    AmbiguousSource,
    /// Invalid passthrough device name: {0}
    InvalidDeviceName(String),
    /// Invalid passthrough group path: {0}
    InvalidGroupPath(String),
}

fn serialize_sbdf_as_str<S: Serializer>(
    sbdf: &Option<PciSBDF>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Serialize to string to present in a "0000:01:02.03" format
    match sbdf {
        Some(sbdf) => serializer.collect_str(sbdf),
        None => serializer.serialize_none(),
    }
}

fn deserialize_sbdf_from_str<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<PciSBDF>, D::Error> {
    let s = String::deserialize(deserializer)?;
    PciSBDF::new_from_str(&s).map(Some).ok_or_else(|| {
        serde::de::Error::custom(DevicePassthroughConfigError::InvalidSBDF(s.to_string()))
    })
}

/// Config for device passthrough
///
/// A device is named either by its host PCI address (`sbdf`, opened through
/// sysfs), or by its VFIO group node and device name (`group_path` plus
/// `device`, a PCI address or an mdev UUID). The second form needs no sysfs,
/// so it works inside a jail.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePassthroughConfig {
    /// ID of the device
    pub id: String,
    /// Host identifier for the PCI device
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_sbdf_as_str",
        deserialize_with = "deserialize_sbdf_from_str"
    )]
    pub sbdf: Option<PciSBDF>,
    /// Path of the VFIO group character device
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_path: Option<PathBuf>,
    /// Name of the device within its VFIO group
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Fail the attach unless the device supports stop-and-copy migration
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub require_migration: bool,
}

/// Where a passthrough device is opened from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePassthroughSource<'a> {
    /// A host PCI device, found through sysfs.
    Sysfs(PciSBDF),
    /// A device of an explicit VFIO group node.
    Group {
        /// The group's character device node.
        group_path: &'a Path,
        /// The device name within the group.
        device: &'a str,
    },
}

impl std::fmt::Display for DevicePassthroughSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sysfs(sbdf) => write!(f, "{sbdf}"),
            Self::Group { group_path, device } => {
                write!(f, "{}/{device}", group_path.display())
            }
        }
    }
}

impl DevicePassthroughConfig {
    /// Validate the configuration and describe where the device comes from.
    pub fn source(&self) -> Result<DevicePassthroughSource<'_>, DevicePassthroughConfigError> {
        match (&self.sbdf, &self.group_path, &self.device) {
            (Some(sbdf), None, None) => Ok(DevicePassthroughSource::Sysfs(*sbdf)),
            (None, Some(group_path), Some(device)) => {
                if device.is_empty()
                    || device.len() > 64
                    || !device
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-:._".contains(&b))
                {
                    return Err(DevicePassthroughConfigError::InvalidDeviceName(
                        device.clone(),
                    ));
                }
                if !group_path.is_absolute() {
                    return Err(DevicePassthroughConfigError::InvalidGroupPath(
                        group_path.display().to_string(),
                    ));
                }
                Ok(DevicePassthroughSource::Group {
                    group_path,
                    device,
                })
            }
            _ => Err(DevicePassthroughConfigError::AmbiguousSource),
        }
    }
}

/// Configs for device passthrough
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DevicePassthroughConfigs {
    /// configs
    pub configs: Vec<DevicePassthroughConfig>,
}

impl DevicePassthroughConfigs {
    /// Add config to the set. Overwrite existing one if
    /// ids are same.
    pub fn add(
        &mut self,
        config: DevicePassthroughConfig,
    ) -> Result<(), DevicePassthroughConfigError> {
        let source = config.source()?;
        for existing in self.configs.iter().filter(|b| b.id != config.id) {
            if existing.source()? == source {
                return Err(DevicePassthroughConfigError::DuplicateSource(
                    source.to_string(),
                ));
            }
        }
        if let Some(old_config) = self.configs.iter_mut().find(|b| b.id == config.id) {
            *old_config = config;
        } else {
            self.configs.push(config);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_device_passthrough_config_and_overwrite() {
        let id1 = PciSBDF::new_from_str("01:00.0").unwrap();
        let id2 = PciSBDF::new_from_str("02:00.0").unwrap();

        let mut configs = DevicePassthroughConfigs::default();

        configs
            .add(DevicePassthroughConfig {
                id: "dev0".to_string(),
                sbdf: Some(id1),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(configs.configs.len(), 1);
        assert_eq!(configs.configs[0].sbdf, Some(id1));

        configs
            .add(DevicePassthroughConfig {
                id: "dev0".to_string(),
                sbdf: Some(id2),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(configs.configs.len(), 1);
        assert_eq!(configs.configs[0].sbdf, Some(id2));

        configs
            .add(DevicePassthroughConfig {
                id: "dev1".to_string(),
                sbdf: Some(id1),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(configs.configs.len(), 2);
        assert_eq!(configs.configs[0].sbdf, Some(id2));
        assert_eq!(configs.configs[1].sbdf, Some(id1));

        configs
            .add(DevicePassthroughConfig {
                id: "dev1".to_string(),
                sbdf: Some(id2),
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(configs.configs.len(), 2);
        assert_eq!(configs.configs[0].sbdf, Some(id2));
        assert_eq!(configs.configs[1].sbdf, Some(id1));
    }
}

#[cfg(test)]
mod group_source_tests {
    use super::*;

    fn group_config(id: &str, path: &str, device: &str) -> DevicePassthroughConfig {
        DevicePassthroughConfig {
            id: id.to_string(),
            group_path: Some(PathBuf::from(path)),
            device: Some(device.to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn test_group_source_requires_exactly_one_form() {
        let sbdf = PciSBDF::new_from_str("01:00.0").unwrap();
        let mut both = group_config("gpu0", "/dev/vfio/gpu0", "0000:01:00.0");
        both.sbdf = Some(sbdf);
        assert!(matches!(
            both.source(),
            Err(DevicePassthroughConfigError::AmbiguousSource)
        ));
        let neither = DevicePassthroughConfig {
            id: "gpu0".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            neither.source(),
            Err(DevicePassthroughConfigError::AmbiguousSource)
        ));
        let mut half = group_config("gpu0", "/dev/vfio/gpu0", "x");
        half.device = None;
        half.source().unwrap_err();
    }

    #[test]
    fn test_group_source_rejects_unsafe_names() {
        group_config("gpu0", "/dev/vfio/gpu0", "../../etc")
            .source()
            .unwrap_err();
        group_config("gpu0", "dev/vfio/gpu0", "uuid").source().unwrap_err();
        group_config("gpu0", "/dev/vfio/gpu0", "")
            .source()
            .unwrap_err();
        let uuid = "e7a9c3b0-6b0f-4d8e-9a1c-2f4b5d6e7f80";
        assert_eq!(
            group_config("gpu0", "/dev/vfio/gpu0", uuid).source().unwrap(),
            DevicePassthroughSource::Group {
                group_path: Path::new("/dev/vfio/gpu0"),
                device: uuid,
            }
        );
    }

    #[test]
    fn test_group_sources_are_deduplicated() {
        let mut configs = DevicePassthroughConfigs::default();
        configs
            .add(group_config("gpu0", "/dev/vfio/gpu0", "uuid-a"))
            .unwrap();
        configs
            .add(group_config("gpu1", "/dev/vfio/gpu0", "uuid-a"))
            .unwrap_err();
        configs
            .add(group_config("gpu1", "/dev/vfio/gpu1", "uuid-a"))
            .unwrap();
        // Re-adding an id replaces its source.
        configs
            .add(group_config("gpu0", "/dev/vfio/gpu2", "uuid-b"))
            .unwrap();
        assert_eq!(configs.configs.len(), 2);
        assert_eq!(configs.configs[0].device.as_deref(), Some("uuid-b"));
    }

    #[test]
    fn test_group_source_json_round_trip() {
        let json = r#"{"id":"gpu0","group_path":"/dev/vfio/gpu0","device":"uuid-a","require_migration":true}"#;
        let config: DevicePassthroughConfig = serde_json::from_str(json).unwrap();
        assert!(config.require_migration);
        assert_eq!(serde_json::to_string(&config).unwrap(), json);
        let legacy: DevicePassthroughConfig =
            serde_json::from_str(r#"{"id":"dev","sbdf":"0000:00:1f.0"}"#).unwrap();
        assert_eq!(
            serde_json::to_string(&legacy).unwrap(),
            r#"{"id":"dev","sbdf":"0000:00:1f.0"}"#
        );
    }
}
