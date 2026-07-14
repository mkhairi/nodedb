// SPDX-License-Identifier: BUSL-1.1

//! Integration tests: table-qualified column refs in UPDATE/DELETE WHERE
//! clauses resolve exactly like their bare-column form.
//!
//! libpq-based ORMs (e.g. Diesel) always emit `WHERE "table"."col" = ...`;
//! before the qualifier normalization in `plan_update`/`plan_delete` these
//! statements silently matched zero rows. A qualifier naming a different
//! table must error rather than match nothing.

use nodedb_sql::types::{CollectionInfo, EngineType};
use nodedb_sql::{SqlCatalog, SqlCatalogError, SqlError, plan_sql};
use nodedb_types::DatabaseId;

struct Catalog;

impl SqlCatalog for Catalog {
    fn get_collection(
        &self,
        _: DatabaseId,
        name: &str,
    ) -> std::result::Result<Option<CollectionInfo>, SqlCatalogError> {
        let info = match name {
            "users" => Some(CollectionInfo {
                name: "users".into(),
                engine: EngineType::DocumentSchemaless,
                columns: Vec::new(),
                primary_key: Some("id".into()),
                has_auto_tier: false,
                indexes: Vec::new(),
                bitemporal: false,
                primary: nodedb_types::PrimaryEngine::Document,
                vector_primary: None,
                partition_strategy: nodedb_types::PartitionStrategy::CollectionHomed,
            }),
            _ => None,
        };
        Ok(info)
    }

    fn lookup_array(&self, _name: &str) -> Option<nodedb_sql::types::ArrayCatalogView> {
        None
    }

    fn array_exists(&self, _name: &str) -> bool {
        false
    }
}

/// The qualified form must plan identically to the bare form.
fn assert_plans_match(qualified: &str, bare: &str) {
    let q = plan_sql(qualified, &Catalog).unwrap_or_else(|e| {
        panic!("qualified statement failed to plan: {qualified}: {e:?}")
    });
    let b = plan_sql(bare, &Catalog).unwrap();
    assert_eq!(
        format!("{q:?}"),
        format!("{b:?}"),
        "qualified plan differs from bare plan\n  qualified: {qualified}\n  bare: {bare}"
    );
}

#[test]
fn update_where_table_qualified() {
    assert_plans_match(
        r#"UPDATE "users" SET "karma" = 7 WHERE ("users"."id" = 2)"#,
        "UPDATE users SET karma = 7 WHERE id = 2",
    );
}

#[test]
fn update_where_alias_qualified() {
    assert_plans_match(
        "UPDATE users AS u SET karma = 7 WHERE u.id = 2",
        "UPDATE users SET karma = 7 WHERE id = 2",
    );
}

#[test]
fn delete_where_table_qualified() {
    assert_plans_match(
        r#"DELETE FROM "users" WHERE ("users"."karma" < 10)"#,
        "DELETE FROM users WHERE karma < 10",
    );
}

#[test]
fn delete_where_alias_qualified() {
    assert_plans_match(
        "DELETE FROM users AS u WHERE u.id = 1",
        "DELETE FROM users WHERE id = 1",
    );
}

#[test]
fn update_where_foreign_qualifier_errors() {
    let result = plan_sql("UPDATE users SET karma = 7 WHERE orders.id = 2", &Catalog);
    assert!(
        matches!(result, Err(SqlError::UnknownTable { .. }) | Err(SqlError::Unsupported { .. })),
        "foreign qualifier must error, got {result:?}"
    );
}

#[test]
fn delete_where_foreign_qualifier_errors() {
    let result = plan_sql("DELETE FROM users WHERE orders.id = 2", &Catalog);
    assert!(
        matches!(result, Err(SqlError::UnknownTable { .. }) | Err(SqlError::Unsupported { .. })),
        "foreign qualifier must error, got {result:?}"
    );
}
