"""
クエリ拡張モジュールのユニットテスト

このテストスイートは、クエリ拡張機能の正常動作と
エラーハンドリングを検証します。

テスト観点:
    - 正常系: 各カテゴリのクエリが正しく拡張されるか
    - 境界値: 空クエリ、非常に長いクエリ
    - 異常系: API障害時のフォールバック動作
    - 回帰テスト: 元の失敗ケースが修正されるか

実行方法:
    pytest tests/test_query_expander.py -v

    # 特定のテストのみ実行
    pytest tests/test_query_expander.py::TestQueryExpander::test_expand_attendance_query -v

    # カバレッジ付き
    pytest tests/test_query_expander.py --cov=query_expander --cov-report=term-missing
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query_expander import QueryExpander, ExpansionResult


class TestExpansionResult:
    """ExpansionResult データクラスのテスト"""

    def test_create_successful_result(self):
        """成功時のExpansionResultを正しく作成できる"""
        result = ExpansionResult(
            original_query="有給の申請方法は？",
            expanded_query="有給の申請方法は？ 有給休暇 勤怠",
            added_keywords=["有給休暇", "勤怠"],
            expansion_applied=True
        )

        assert result.original_query == "有給の申請方法は？"
        assert result.expanded_query == "有給の申請方法は？ 有給休暇 勤怠"
        assert result.added_keywords == ["有給休暇", "勤怠"]
        assert result.expansion_applied is True
        assert result.error_message is None

    def test_create_failed_result(self):
        """失敗時のExpansionResultを正しく作成できる"""
        result = ExpansionResult(
            original_query="テストクエリ",
            expanded_query="テストクエリ",  # フォールバック
            added_keywords=[],
            expansion_applied=False,
            error_message="API Error"
        )

        assert result.original_query == "テストクエリ"
        assert result.expanded_query == "テストクエリ"
        assert result.added_keywords == []
        assert result.expansion_applied is False
        assert result.error_message == "API Error"


class TestQueryExpander:
    """QueryExpander クラスのテスト"""

    def test_init_default_values(self):
        """デフォルト値で初期化できる"""
        expander = QueryExpander()

        assert expander.enabled is True
        assert expander.temperature == 0.1
        assert expander.max_keywords == 5

    def test_init_custom_values(self):
        """カスタム値で初期化できる"""
        expander = QueryExpander(
            enabled=False,
            temperature=0.5,
            max_keywords=3
        )

        assert expander.enabled is False
        assert expander.temperature == 0.5
        assert expander.max_keywords == 3

    def test_empty_query_raises_error(self):
        """空のクエリはValueErrorを発生させる"""
        expander = QueryExpander()

        with pytest.raises(ValueError, match="クエリが空です"):
            expander.expand("")

    def test_whitespace_only_query_raises_error(self):
        """空白のみのクエリはValueErrorを発生させる"""
        expander = QueryExpander()

        with pytest.raises(ValueError, match="クエリが空です"):
            expander.expand("   ")

    def test_disabled_expander_returns_original(self):
        """無効化時は元のクエリをそのまま返す"""
        expander = QueryExpander(enabled=False)
        result = expander.expand("テストクエリ")

        assert result.expansion_applied is False
        assert result.expanded_query == "テストクエリ"
        assert result.added_keywords == []
        assert result.error_message == "Query expansion is disabled"

    def test_build_prompt(self):
        """プロンプトが正しく構築される"""
        expander = QueryExpander(max_keywords=5)
        prompt = expander._build_prompt("有給の申請方法は？")

        # プロンプトに必要な要素が含まれているか確認
        assert "有給の申請方法は？" in prompt
        assert "5" in prompt  # max_keywords
        assert "勤怠管理" in prompt  # ドメイン知識
        assert "経費精算" in prompt
        assert "ITサポート" in prompt

    def test_parse_keywords_comma_separated(self):
        """カンマ区切りのキーワードを正しくパースできる"""
        expander = QueryExpander(max_keywords=5)
        keywords = expander._parse_keywords("有給休暇, 勤怠, 休み")

        assert keywords == ["有給休暇", "勤怠", "休み"]

    def test_parse_keywords_newline_separated(self):
        """改行区切りのキーワードを正しくパースできる"""
        expander = QueryExpander(max_keywords=5)
        keywords = expander._parse_keywords("有給休暇\n勤怠\n休み")

        assert keywords == ["有給休暇", "勤怠", "休み"]

    def test_parse_keywords_mixed_separators(self):
        """混合の区切りでもパースできる"""
        expander = QueryExpander(max_keywords=5)
        keywords = expander._parse_keywords("有給休暇, 勤怠\n休み")

        assert keywords == ["有給休暇", "勤怠", "休み"]

    def test_parse_keywords_empty_response(self):
        """空のレスポンスは空リストを返す"""
        expander = QueryExpander()
        keywords = expander._parse_keywords("")

        assert keywords == []

    def test_parse_keywords_respects_max(self):
        """max_keywordsを超えないようにトリミングされる"""
        expander = QueryExpander(max_keywords=2)
        keywords = expander._parse_keywords("有給休暇, 勤怠, 休み, 申請")

        assert len(keywords) == 2
        assert keywords == ["有給休暇", "勤怠"]

    def test_parse_keywords_with_explanation(self):
        """括弧注釈混入パターン: 括弧を除去してキーワードは残す"""
        expander = QueryExpander()
        response = "有給休暇, 勤怠（勤怠カテゴリです）, 休み"
        result = expander._parse_keywords(response)
        assert result == ["有給休暇", "勤怠", "休み"]

    def test_parse_keywords_json_format(self):
        """JSON配列が返った場合に正しくパースできる"""
        expander = QueryExpander()
        response = '["有給休暇", "勤怠", "休み"]'
        result = expander._parse_keywords(response)
        assert result == ["有給休暇", "勤怠", "休み"]

    def test_parse_keywords_json_fallback(self):
        """JSONが壊れていた場合、従来方式にフォールバック（最低限キーワードは抽出）"""
        expander = QueryExpander()
        response = '["有給休暇", 勤怠'  # 壊れたJSON
        result = expander._parse_keywords(response)
        # フォールバックでカンマ分割 → クォート除去で「有給休暇」は抽出される
        assert "有給休暇" in result

    @patch('query_expander.genai.GenerativeModel')
    def test_expand_success(self, mock_model_class):
        """正常なクエリ拡張が成功する"""
        # モックのセットアップ
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "有給休暇, 勤怠, 休み"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        expander = QueryExpander()
        result = expander.expand("有給の申請方法は？")

        assert result.expansion_applied is True
        assert "有給の申請方法は？" in result.expanded_query
        assert "有給休暇" in result.added_keywords
        assert result.error_message is None

    @patch('query_expander.genai.GenerativeModel')
    def test_expand_api_failure_fallback(self, mock_model_class):
        """API障害時は元のクエリで継続する"""
        # モックのセットアップ - 例外を発生させる
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model

        expander = QueryExpander()
        result = expander.expand("テストクエリ")

        # フォールバック動作の確認
        assert result.expansion_applied is False
        assert result.expanded_query == "テストクエリ"  # 元のクエリのまま
        assert result.added_keywords == []
        assert "API Error" in result.error_message

    @patch('query_expander.genai.GenerativeModel')
    def test_expand_empty_keywords_fallback(self, mock_model_class):
        """キーワードが抽出できない場合も元のクエリで継続する"""
        # モックのセットアップ - 空のレスポンス
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        expander = QueryExpander()
        result = expander.expand("テストクエリ")

        assert result.expansion_applied is False
        assert result.expanded_query == "テストクエリ"
        assert result.added_keywords == []


class TestIntegration:
    """
    統合テスト - 実際のAPIを使用

    注意: これらのテストは実際のGemini APIを呼び出すため、
    CI/CDでは環境変数がないとスキップされます。

    実行方法:
        # 環境変数を設定してから実行
        export CHROMA_GOOGLE_GENAI_API_KEY=your_api_key
        pytest tests/test_query_expander.py::TestIntegration -v
    """

    @pytest.fixture
    def expander(self):
        """QueryExpanderのフィクスチャ"""
        api_key = os.getenv("CHROMA_GOOGLE_GENAI_API_KEY")
        if not api_key:
            pytest.skip("CHROMA_GOOGLE_GENAI_API_KEY not set")
        return QueryExpander()

    def test_expand_attendance_query(self, expander):
        """勤怠関連のクエリが正しく拡張される"""
        result = expander.expand("有給の申請方法は？")

        assert result.expansion_applied is True
        assert len(result.added_keywords) > 0
        # 勤怠関連のキーワードが含まれることを期待
        keywords_str = " ".join(result.added_keywords).lower()
        assert any(kw in keywords_str for kw in ["有給", "勤怠", "休暇", "休み"])

    def test_expand_expense_query(self, expander):
        """経費関連のクエリが正しく拡張される"""
        result = expander.expand("経費精算の締め切りは？")

        assert result.expansion_applied is True
        assert len(result.added_keywords) > 0
        # 経費関連のキーワードが含まれることを期待
        keywords_str = " ".join(result.added_keywords).lower()
        assert any(kw in keywords_str for kw in ["経費", "精算", "expense", "領収"])

    def test_expand_it_query(self, expander):
        """IT関連のクエリが正しく拡張される"""
        result = expander.expand("業務用ソフトのインストール方法")

        assert result.expansion_applied is True
        assert len(result.added_keywords) > 0
        # IT関連のキーワードが含まれることを期待
        keywords_str = " ".join(result.added_keywords).lower()
        assert any(kw in keywords_str for kw in ["it", "ソフトウェア", "インストール", "pc"])


class TestRegressionCases:
    """
    回帰テスト - 元の問題ケースが改善されることを確認

    これらのテストは、元の問題（一般語が類似度を支配する）が
    クエリ拡張によって改善されることを確認します。

    注意: 実際のChromaDB検索を含む統合テストは別途実施が必要です。
    ここではクエリ拡張の出力のみをテストします。
    """

    @pytest.fixture
    def expander(self):
        """QueryExpanderのフィクスチャ"""
        api_key = os.getenv("CHROMA_GOOGLE_GENAI_API_KEY")
        if not api_key:
            pytest.skip("CHROMA_GOOGLE_GENAI_API_KEY not set")
        return QueryExpander()

    def test_yuukyuu_query_adds_attendance_keywords(self, expander):
        """
        有給の申請方法 → 勤怠関連キーワードが追加される

        元の問題:
            「申請」がit_support.mdのソフトウェア申請にマッチしてしまう

        期待:
            「有給休暇」「勤怠」などのキーワードが追加され、
            attendance.mdが上位にくるようになる
        """
        result = expander.expand("有給の申請方法は？")

        # 拡張されていること
        assert result.expansion_applied is True
        assert len(result.expanded_query) > len("有給の申請方法は？")

        # IT関連ではなく勤怠関連のキーワードが追加されていること
        keywords_lower = [kw.lower() for kw in result.added_keywords]
        # 「it」「ソフトウェア」などのIT関連キーワードが含まれていないこと
        assert not any("it" in kw or "ソフトウェア" in kw for kw in keywords_lower)

    def test_keihi_query_adds_expense_keywords(self, expander):
        """
        経費精算の締め切り → 経費関連キーワードが追加される

        元の問題:
            「締め切り」「申請」がit_support.mdや勤怠にマッチしてしまう

        期待:
            「経費」「精算」などのキーワードが追加され、
            expense.mdが上位にくるようになる
        """
        result = expander.expand("経費精算の締め切りは？")

        assert result.expansion_applied is True
        assert len(result.expanded_query) > len("経費精算の締め切りは？")

        # 経費関連のキーワードが追加されていること
        keywords_str = " ".join(result.added_keywords)
        assert any(kw in keywords_str for kw in ["経費", "精算", "領収", "expense"])

    def test_install_query_adds_it_keywords(self, expander):
        """
        業務用ソフトのインストール方法 → IT関連キーワードが追加される

        元の問題:
            「方法」「業務」がexpense.mdにマッチしてしまう

        期待:
            「IT」「ソフトウェア」などのキーワードが追加され、
            it_support.mdが上位にくるようになる
        """
        result = expander.expand("業務用ソフトのインストール方法")

        assert result.expansion_applied is True
        assert len(result.expanded_query) > len("業務用ソフトのインストール方法")

        # IT関連のキーワードが追加されていること
        keywords_str = " ".join(result.added_keywords).lower()
        assert any(kw in keywords_str for kw in ["it", "ソフトウェア", "インストール", "pc", "ヘルプ"])


if __name__ == "__main__":
    # コマンドラインから直接実行する場合
    pytest.main([__file__, "-v"])
